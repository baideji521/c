"""反溯主流程（Python API 入口）。

    Final.mp4 + Sources/  ->  segments(source + 起止时间 + confidence)  ->  裁剪导出

链路：
    镜头检测 -> ISC 特征(1FPS, 带缓存) -> 多素材候选检索 -> VCSL/TN 时序对齐
    -> 局部精定位(4FPS) -> 困难样本升级(8FPS + TTA) -> 置信度分级 -> 片段合并 -> FFmpeg 导出

GUI / CLI 都只是本模块的前端；本模块不依赖任何界面。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import VIDEO_EXTS, SourceTraceConfig
from .export.csv_report import write_csv
from .export.ffmpeg import ClipResult, clip_name, concat_clips, export_clip
from .export.json_report import build_result, write_errors, write_json
from .export.visualize import make_verification
from .features.cache import FeatureStore
from .features.extractor import TTASpec, build_extractor
from .refinement.fine_match import refine_segment
from .retrieval.candidates import ShotCandidates, candidates_for_shots
from .retrieval.index import build_index_from_dir
from .temporal.alignment import align
from .temporal.mapping import (
    TimeSegment,
    compute_confidence,
    decide_status,
    merge_segments,
    path_to_time,
)
from .utils.device import peak_vram_mb
from .utils.log import Timer, get_logger
from .video.probe import VideoInfo, probe, scan_videos
from .video.shot_detector import detect_shots


@dataclass
class TraceResult:
    output_dir: Path
    segments: list[TimeSegment] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    clips: list[ClipResult] = field(default_factory=list)
    verification: list[dict] = field(default_factory=list)
    json_path: Path | None = None
    csv_path: Path | None = None


def run_trace(
    query: Path | str,
    sources: Path | str,
    output: Path | str,
    cfg: SourceTraceConfig | None = None,
    export_clips: bool = True,
    verify: bool = True,
    progress=None,
) -> TraceResult:
    """执行一次完整反溯。

    progress: 可选回调 ``progress(stage: str, done: int, total: int)``，供 GUI 显示进度。
    """
    log = get_logger()
    cfg = cfg or SourceTraceConfig()
    timer = Timer()
    query_path = Path(query)
    sources_dir = Path(sources)
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []

    def _tick(stage: str, done: int, total: int) -> None:
        if progress is not None:
            try:
                progress(stage, done, total)
            except Exception:
                pass

    # ---------------- 1. 输入探测 ----------------
    with timer.stage("probe"):
        q_info = probe(query_path)
        log.info("成片：%s", q_info.describe())
        log.info("扫描原始素材...")
        source_files = scan_videos(sources_dir, VIDEO_EXTS)
        # 成片可能就放在素材目录里（例如 test_pic/a.mp4 与 b1..b5.mp4 同级），
        # 若不剔除，成片会和自己完美匹配，结果全部指向成片本身。
        q_resolved = query_path.resolve()
        skipped_self = [p for p in source_files if p.resolve() == q_resolved]
        if skipped_self:
            source_files = [p for p in source_files if p.resolve() != q_resolved]
            log.info("素材目录中包含成片本身，已自动跳过：%s", skipped_self[0].name)
        if not source_files:
            raise RuntimeError(f"素材目录中没有找到视频文件：{sources_dir}")
        log.info("找到 %d 个视频", len(source_files))
    _tick("probe", 1, 1)

    # ---------------- 2. 特征提取器 ----------------
    with timer.stage("model_init"):
        extractor = build_extractor(
            cfg.feature,
            tta=TTASpec(enabled=False, scales=cfg.refinement.tta_scales, hflip=cfg.refinement.tta_hflip),
        )
        log.info(
            "描述子：%s  维度=%d  modality=%s  归一化=%s  权重=%s",
            extractor.meta.get("model"), extractor.meta.get("dim"),
            extractor.meta.get("modality"), extractor.meta.get("normalization"),
            extractor.meta.get("weights"),
        )
    store = FeatureStore(cfg.cache_dir(out_dir.parent), cfg.cache.enabled, cfg.cache.full_hash)

    # ---------------- 3. 素材索引（粗，1FPS）----------------
    with timer.stage("feature_sources"):
        index, src_errors = build_index_from_dir(
            source_files,
            store,
            extractor,
            fps=cfg.sampling.coarse_fps,
            frame_size=cfg.sampling.frame_size,
            backend=cfg.retrieval.backend,
        )
        errors.extend(src_errors)
    _tick("feature_sources", index.size, len(source_files))

    # ---------------- 4. 成片特征 + 镜头检测 ----------------
    with timer.stage("feature_query"):
        q_features = store.get_or_build(
            q_info, extractor, fps=cfg.sampling.coarse_fps, frame_size=cfg.sampling.frame_size, tag="isc"
        )
    with timer.stage("shot_detect"):
        shots = detect_shots(q_info, cfg.shot)
    _tick("shot_detect", len(shots), len(shots))

    # ---------------- 5. 候选检索 ----------------
    log.info("开始候选检索...")
    with timer.stage("retrieval"):
        shot_cands = candidates_for_shots(index, q_features, shots, cfg.retrieval)

    # ---------------- 6. 时序对齐 + 精定位 ----------------
    log.info("开始时间对齐...")
    segments: list[TimeSegment] = []
    source_infos: dict[int, VideoInfo] = {}
    for n, sc in enumerate(shot_cands, 1):
        try:
            seg = _resolve_shot(cfg, store, extractor, index, q_info, sc, source_infos, timer)
        except Exception as exc:
            log.error("片段 %d 处理失败：%s: %s", sc.shot.shot_id, type(exc).__name__, exc)
            errors.append(
                {
                    "segment": sc.shot.shot_id,
                    "query_start": sc.shot.start,
                    "query_end": sc.shot.end,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            seg = TimeSegment(
                id=sc.shot.shot_id,
                query_start=sc.shot.start,
                query_end=sc.shot.end,
                source=None,
                source_id=None,
                source_start=None,
                source_end=None,
                confidence=0.0,
                status="UNKNOWN",
                scores={},
                candidates=sc.candidates,
                method=cfg.alignment.method,
            )
        segments.append(seg)
        _tick("align", n, len(shot_cands))

    segments = merge_segments(segments)
    for s in segments:
        if s.status == "UNKNOWN":
            log.info("片段 %02d [%.2f-%.2f] 未能可靠定位 -> UNKNOWN", s.id, s.query_start, s.query_end)
        else:
            log.info(
                "片段 %02d [%.2f-%.2f] -> %s [%.3f-%.3f] speed=%.3f confidence=%.3f (%s)",
                s.id, s.query_start, s.query_end, s.source, s.source_start, s.source_end,
                s.speed, s.confidence, s.status,
            )
    log.info("定位完成：%d 个片段（HIGH=%d MEDIUM=%d LOW=%d UNKNOWN=%d）",
             len(segments),
             sum(1 for s in segments if s.status == "HIGH"),
             sum(1 for s in segments if s.status == "MEDIUM"),
             sum(1 for s in segments if s.status == "LOW"),
             sum(1 for s in segments if s.status == "UNKNOWN"))

    # ---------------- 7. 报告 ----------------
    meta = {
        "config": cfg.to_dict(),
        "descriptor": extractor.meta,
        "device": extractor.meta.get("device", "cpu"),
        "n_sources_scanned": len(source_files),
        "n_sources_indexed": index.size,
        "shots": [s.to_dict() for s in shots],
    }
    result = build_result(
        query_path.name, q_info.duration, [e.name for e in index.entries], segments, meta
    )
    json_path = write_json(result, out_dir / "result.json")
    csv_path = write_csv(segments, out_dir / "result.csv")
    log.info("报告已输出：%s / %s", json_path.name, csv_path.name)

    # ---------------- 8. 裁剪导出 ----------------
    clips: list[ClipResult] = []
    if export_clips:
        log.info("正在导出原始片段...")
        with timer.stage("export"):
            clips_dir = out_dir / "segments"
            for s in segments:
                if s.status == "UNKNOWN" or s.source is None or s.source_start is None:
                    continue
                src = sources_dir / s.source
                name = clip_name(s.id, s.source, s.source_start, s.source_end or s.source_start)
                res = export_clip(src, s.source_start, s.source_end or s.source_start, clips_dir / name, cfg.export)
                clips.append(res)
                if not res.ok:
                    errors.append({"segment": s.id, "stage": "export", "error": res.error})
                if cfg.export.export_query_clips:
                    export_clip(
                        query_path,
                        s.query_start,
                        s.query_end,
                        clips_dir / f"{s.id:03d}_FINAL_{s.query_start:.2f}-{s.query_end:.2f}.mp4",
                        cfg.export,
                    )
        log.info("已导出 %d 个原始片段 -> %s", sum(1 for c in clips if c.ok), out_dir / "segments")

        # 把定位出来的原始片段按成片顺序拼成一条新素材
        if cfg.export.merge_recovered:
            ok_paths = [c.path for c in clips if c.ok]
            if len(ok_paths) >= 1:
                with timer.stage("merge"):
                    merged = concat_clips(
                        ok_paths, out_dir / "recovered_from_sources.mp4", cfg.export,
                        width=q_info.width, height=q_info.height, fps=q_info.fps,
                    )
                if merged.ok:
                    clips.append(merged)
                else:
                    errors.append({"stage": "merge", "error": merged.error})

    # ---------------- 9. 可视化验证 ----------------
    verification: list[dict] = []
    if verify and (cfg.export.contact_sheet or cfg.export.compare_video):
        with timer.stage("verify"):
            try:
                verification = make_verification(
                    query_path, sources_dir, segments, out_dir / "verification", cfg.export
                )
            except Exception as exc:
                log.warning("验证材料生成失败：%s: %s", type(exc).__name__, exc)
                errors.append({"stage": "verify", "error": f"{type(exc).__name__}: {exc}"})

    write_errors(errors, out_dir / "errors.json")
    timings = timer.as_dict()
    timings["peak_vram_mb"] = round(peak_vram_mb(), 1)
    result["timings"] = timings
    write_json(result, out_dir / "result.json")
    log.info("总耗时 %.2fs：%s", timings["total"], timings)

    return TraceResult(
        output_dir=out_dir,
        segments=segments,
        result=result,
        errors=errors,
        timings=timings,
        clips=clips,
        verification=verification,
        json_path=json_path,
        csv_path=csv_path,
    )


# ---------------------------------------------------------------- 单片段求解


def _composite(d: dict) -> float:
    """视觉相似度 × 时序覆盖率的复合分：抑制「相似度高但只对上几帧」的伪路径。"""
    return float(d["visual"]) * (0.5 + 0.5 * float(d["coverage"]))


def _align_candidate(
    cfg: SourceTraceConfig,
    cand: dict,
    entry,
    query_desc: np.ndarray,
    query_ts: np.ndarray,
    source_desc: np.ndarray,
    source_ts: np.ndarray,
    shot,
    fps: float,
) -> dict | None:
    """在给定采样密度下把一个候选素材与镜头做时序对齐。"""
    sim = query_desc.astype(np.float32) @ source_desc.astype(np.float32).T
    paths = align(sim, cfg.alignment, max_paths=3)
    if not paths:
        return None
    path = paths[0]
    s_start, s_end, fit = path_to_time(
        path, query_ts, source_ts, shot.start, shot.end, cfg.alignment, entry.duration
    )
    item = {
        "cand": cand,
        "entry": entry,
        "path": path,
        "s_start": s_start,
        "s_end": s_end,
        "fit": fit,
        "visual": path.visual,
        "coverage": path.coverage,
        "align_fps": fps,
    }
    item["score"] = _composite(item)
    return item


def _crop_compensate(
    cfg: SourceTraceConfig,
    store: FeatureStore,
    extractor,
    index,
    q_info: VideoInfo,
    sc: ShotCandidates,
    source_infos: dict[int, VideoInfo],
    timer: Timer,
) -> tuple[dict, float] | None:
    """裁边补偿：把原素材中心裁切到与被裁成片相同的视野后重新对齐。

    成片被裁掉画面边缘后，全局描述子（ISC）会显著漂移——实测 10% 裁切会让真值区间的
    相似度低于同素材其他时间点的伪匹配，于是一条合法时序链都建不出来。把原素材按
    ``refinement.crop_views`` 的比例做中心裁切再比对，可以把视野重新对上。

    代价较高（每个比例都要对候选素材全片重提特征，结果会进缓存），
    因此只在片段已经判成 UNKNOWN 时才调用。
    """
    log = get_logger()
    shot = sc.shot
    fps = cfg.sampling.fine_fps
    cands = sc.candidates[: max(1, int(cfg.refinement.crop_candidates))]
    items: list[dict] = []

    with timer.stage("refine_crop"):
        # 成片本身已经被裁过，所以 query 侧不裁
        q_fs = store.build_segment(
            q_info, extractor, fps=fps, frame_size=cfg.sampling.frame_size,
            start=shot.start, end=shot.end,
        )
        if len(q_fs) == 0:
            return None
        try:
            for ratio in cfg.refinement.crop_views:
                if ratio >= 1.0:
                    continue
                extractor.set_center_crop(ratio)
                tag = f"isc{fps:g}c{int(round(ratio * 100))}"
                for cand in cands:
                    entry = index.get(cand["source_id"])
                    if entry.source_id not in source_infos:
                        source_infos[entry.source_id] = probe(entry.path)
                    s_info = source_infos[entry.source_id]
                    if s_info.duration > cfg.sampling.rescan_max_source_sec:
                        continue
                    s_fs = store.get_or_build(
                        s_info, extractor, fps=fps, frame_size=cfg.sampling.frame_size, tag=tag
                    )
                    item = _align_candidate(
                        cfg, cand, entry, q_fs.descriptors, q_fs.timestamps,
                        s_fs.descriptors, s_fs.timestamps, shot, fps,
                    )
                    if item is not None:
                        item["crop"] = float(ratio)
                        items.append(item)
        finally:
            extractor.set_center_crop(1.0)

    if not items:
        return None
    items.sort(key=lambda d: -d["score"])
    best = items[0]
    sid = best["entry"].source_id
    others = [d["score"] for d in items[1:] if d["entry"].source_id != sid]
    margin = float(best["score"] - (max(others) if others else 0.0))
    log.debug("裁边补偿候选：%s", [(d["entry"].name, d["crop"], round(d["score"], 3)) for d in items[:4]])
    return best, margin


def _resolve_shot(
    cfg: SourceTraceConfig,
    store: FeatureStore,
    extractor,
    index,
    q_info: VideoInfo,
    sc: ShotCandidates,
    source_infos: dict[int, VideoInfo],
    timer: Timer,
) -> TimeSegment:
    """对单个镜头：粗对齐 -> 中密度补救 -> 选定候选 -> 精定位 -> 置信度分级。"""
    log = get_logger()
    shot = sc.shot
    coarse: list[dict] = []

    with timer.stage("align_coarse"):
        for cand in sc.candidates:
            entry = index.get(cand["source_id"])
            item = _align_candidate(
                cfg, cand, entry, sc.query_desc, sc.query_ts,
                entry.features.descriptors, entry.features.timestamps,
                shot, cfg.sampling.coarse_fps,
            )
            if item is not None:
                coarse.append(item)

    # ---- 逐级加密的全片重扫 ----
    # 1FPS 粗定位与真实剪辑点最多相差 0.5s 相位，短镜头还会因采样点不足凑不出时序链，
    # 结果是「真实来源一条路径都找不到」→ 被别的素材抢走。因此在粗定位不自信时，
    # 依次用 medium_fps / fine_fps 对候选素材做全片重扫（结果进缓存），一旦足够自信就停。
    # 归属于「多级密度精定位」能力，与 refinement 一起开关（benchmark 组合 A/B/C 不启用）。
    if cfg.refinement.enabled:
        levels = [
            f for f in dict.fromkeys((cfg.sampling.medium_fps, cfg.sampling.fine_fps))
            if f > cfg.sampling.coarse_fps
        ]
        for lvl in levels:
            best_score = max((c["score"] for c in coarse), default=0.0)
            if best_score >= cfg.confidence.high:
                break
            tag = f"isc{lvl:g}"
            with timer.stage("align_rescan"):
                q_re = store.build_segment(
                    q_info, extractor, fps=lvl, frame_size=cfg.sampling.frame_size,
                    start=shot.start, end=shot.end,
                )
                if len(q_re) == 0:
                    continue
                for cand in sc.candidates:
                    entry = index.get(cand["source_id"])
                    if entry.source_id not in source_infos:
                        source_infos[entry.source_id] = probe(entry.path)
                    s_info = source_infos[entry.source_id]
                    if s_info.duration > cfg.sampling.rescan_max_source_sec:
                        log.debug("素材 %s 时长 %.0fs 超过重扫上限，跳过 %.1fFPS 全片重扫",
                                  entry.name, s_info.duration, lvl)
                        continue
                    s_re = store.get_or_build(
                        s_info, extractor, fps=lvl, frame_size=cfg.sampling.frame_size, tag=tag,
                    )
                    item = _align_candidate(
                        cfg, cand, entry, q_re.descriptors, q_re.timestamps,
                        s_re.descriptors, s_re.timestamps, shot, lvl,
                    )
                    if item is not None:
                        coarse.append(item)
            log.debug("片段 %02d 以 %.1fFPS 重扫 %d 个候选", shot.shot_id, lvl, len(sc.candidates))

    # 同一素材可能在多个密度下都对齐成功，保留得分更高的一条

    best_per_source: dict[int, dict] = {}
    for item in coarse:
        sid = item["entry"].source_id
        if sid not in best_per_source or item["score"] > best_per_source[sid]["score"]:
            best_per_source[sid] = item
    coarse = list(best_per_source.values())

    if not coarse:

        return TimeSegment(
            id=shot.shot_id,
            query_start=shot.start,
            query_end=shot.end,
            source=None,
            source_id=None,
            source_start=None,
            source_end=None,
            confidence=0.0,
            status="UNKNOWN",
            scores={"reason": 0.0},
            candidates=sc.candidates,
            method=cfg.alignment.method,
        )

    coarse.sort(key=lambda d: -d["score"])
    # 前两名接近时都做精定位，用于计算真实 margin，避免歧义时硬给来源
    to_refine = coarse[:1]
    if len(coarse) > 1 and (_composite(coarse[0]) - _composite(coarse[1])) < 0.12:
        to_refine = coarse[:2]

    refined: list[dict] = []
    for item in to_refine:
        entry = item["entry"]
        if entry.source_id not in source_infos:
            source_infos[entry.source_id] = probe(entry.path)
        s_info = source_infos[entry.source_id]

        best_item = dict(item)
        with timer.stage("refine_fine"):
            r = refine_segment(
                cfg, store, extractor, q_info, s_info,
                shot.start, shot.end, item["s_start"], item["s_end"],
                fps=cfg.sampling.fine_fps, index=index, exclude_source_id=entry.source_id,
            ) if cfg.refinement.enabled else None
        if r is not None:
            best_item.update(
                {"s_start": r.source_start, "s_end": r.source_end, "fit": r.fit,
                 "visual": r.visual, "coverage": r.coverage, "fps": r.fps}
            )
        refined.append(best_item)

    refined.sort(key=lambda d: -_composite(d))
    best = refined[0]
    best_sid = best["entry"].source_id
    # margin 用「视觉 × 时序覆盖」的复合分而不是裸视觉分：
    # 全片重扫会给错误素材也刷出一些视觉分很高但覆盖率很低的伪路径，
    # 若用裸视觉分算 margin，调色/重编码等真实来源相似度被压低的片段会被误判成歧义。
    others = [_composite(d) for d in refined[1:] if d["entry"].source_id != best_sid]
    others += [c["score"] for c in coarse if c["entry"].source_id != best_sid]
    second_score = max(others) if others else 0.0
    margin = float(_composite(best) - second_score)

    conf, scores = compute_confidence(
        visual=best["visual"],
        coverage=best["coverage"],
        margin=margin,
        fit=best["fit"],
        seg_len=shot.duration,
        cfg=cfg.confidence,
    )

    # ---- 困难样本升级：更高帧率 + TTA ----
    if conf < cfg.confidence.enhance_threshold and cfg.refinement.enabled:
        entry = best["entry"]
        s_info = source_infos[entry.source_id]
        used_tta = False
        if cfg.refinement.tta and hasattr(extractor, "set_tta"):
            extractor.set_tta(True)
            used_tta = True
        try:
            with timer.stage("refine_hard"):
                r = refine_segment(
                    cfg, store, extractor, q_info, s_info,
                    shot.start, shot.end, best["s_start"], best["s_end"],
                    fps=cfg.sampling.hard_fps, index=index, exclude_source_id=entry.source_id,
                )
        finally:
            if used_tta:
                extractor.set_tta(False)
        if r is not None:
            conf2, scores2 = compute_confidence(
                visual=r.visual, coverage=r.coverage, margin=margin,
                fit=r.fit, seg_len=shot.duration, cfg=cfg.confidence,
            )
            log.debug("困难样本升级：conf %.3f -> %.3f（%.1fFPS%s）",
                      conf, conf2, cfg.sampling.hard_fps, " + TTA" if used_tta else "")
            if conf2 > conf:
                best.update({"s_start": r.source_start, "s_end": r.source_end, "fit": r.fit,
                             "visual": r.visual, "coverage": r.coverage})
                conf, scores = conf2, scores2

    # ---- 裁边补偿：仍判不出来时，把原素材也裁到相同视野再试一次 ----
    if (
        cfg.refinement.enabled
        and cfg.refinement.crop_compensate
        and hasattr(extractor, "set_center_crop")
        and decide_status(conf, margin, cfg.confidence) == "UNKNOWN"
    ):
        alt = _crop_compensate(cfg, store, extractor, index, q_info, sc, source_infos, timer)
        if alt is not None:
            a_item, a_margin = alt
            conf2, scores2 = compute_confidence(
                visual=a_item["visual"], coverage=a_item["coverage"], margin=a_margin,
                fit=a_item["fit"], seg_len=shot.duration, cfg=cfg.confidence,
            )
            if conf2 > conf:
                log.info("片段 %02d 裁边补偿生效：原素材中心裁切 %.2f，置信度 %.3f -> %.3f",
                         shot.shot_id, a_item["crop"], conf, conf2)
                best, margin, conf, scores = a_item, a_margin, conf2, scores2

    status = decide_status(conf, margin, cfg.confidence)
    entry = best["entry"]
    s_start, s_end = best["s_start"], best["s_end"]
    if s_end - s_start < cfg.alignment.min_segment_sec * 0.5:
        status = "UNKNOWN"

    if status == "UNKNOWN":
        return TimeSegment(
            id=shot.shot_id,
            query_start=shot.start,
            query_end=shot.end,
            source=None,
            source_id=None,
            source_start=None,
            source_end=None,
            confidence=conf,
            status="UNKNOWN",
            scores=scores,
            candidates=sc.candidates,
            method=cfg.alignment.method,
        )

    return TimeSegment(
        id=shot.shot_id,
        query_start=shot.start,
        query_end=shot.end,
        source=entry.name,
        source_id=entry.source_id,
        source_start=s_start,
        source_end=s_end,
        confidence=conf,
        status=status,
        speed=float(best["fit"].slope),
        scores=scores,
        candidates=sc.candidates,
        method=cfg.alignment.method,
    )
