"""自动 Benchmark：多算法组合 × 多测试用例，输出可复现的报告。

用法：
    python -m source_trace.evaluation.benchmark --data testdata --out reports
    python -m source_trace.evaluation.benchmark --data testdata --out reports --configs A,B,C,D --cases basic,multi

算法组合：
    A  ISC only              无时序约束、单候选（每帧取全局最相似）
    B  ISC + Retrieval       多素材 Top-K 候选，仍无时序约束
    C  ISC + VCSL/TN         加入时序对齐
    D  ISC + VCSL + Fine     再加局部高帧率精定位（推荐主线）
    E  D + TTA               困难样本启用 TTA + 分数归一化
    F  D + TransVCL          可选依赖，缺失时记录 FAILURE_REASON 并跳过
    C-dp / C-dtw / C-hv      时序对齐方法横向对比
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import SourceTraceConfig
from ..pipeline import run_trace
from ..utils.log import get_logger, setup_logging
from .ground_truth import GroundTruth, PredSegment
from .metrics import EvalResult, evaluate


@dataclass
class BenchConfig:
    key: str
    title: str
    build: object  # Callable[[], SourceTraceConfig]
    note: str = ""


def _base(model: str) -> SourceTraceConfig:
    cfg = SourceTraceConfig()
    cfg.feature.model = model
    cfg.export.contact_sheet = False
    cfg.export.compare_video = False
    cfg.export.export_query_clips = False
    return cfg


def make_configs(model: str) -> dict[str, BenchConfig]:
    def a():
        c = _base(model)
        c.alignment.method = "none"
        c.refinement.enabled = False
        c.retrieval.top_k = 1
        return c

    def b():
        c = _base(model)
        c.alignment.method = "none"
        c.refinement.enabled = False
        return c

    def cc(method: str):
        def _f():
            c = _base(model)
            c.alignment.method = method
            c.refinement.enabled = False
            return c

        return _f

    def d():
        c = _base(model)
        c.alignment.method = "tn"
        c.refinement.enabled = True
        return c

    def e():
        c = _base(model)
        c.alignment.method = "tn"
        c.refinement.enabled = True
        c.refinement.tta = True
        c.refinement.score_norm = True
        return c

    def f():
        c = _base(model)
        c.alignment.method = "tn"
        c.refinement.enabled = True
        c.refinement.transvcl = True
        return c

    return {
        "A": BenchConfig("A", "ISC only（无时序约束 + 单候选）", a),
        "B": BenchConfig("B", "ISC + Retrieval（多素材 Top-K）", b),
        "C": BenchConfig("C", "ISC + VCSL/TN 时序对齐", cc("tn")),
        "C-dp": BenchConfig("C-dp", "ISC + DP 时序对齐", cc("dp")),
        "C-dtw": BenchConfig("C-dtw", "ISC + DTW 时序对齐", cc("dtw")),
        "C-hv": BenchConfig("C-hv", "ISC + Hough Voting 时序对齐", cc("hv")),
        "D": BenchConfig("D", "ISC + VCSL/TN + 局部精定位（推荐）", d),
        "E": BenchConfig("E", "D + TTA + Score Normalization", e),
        "F": BenchConfig("F", "D + TransVCL 困难样本判别（可选依赖）", f),
    }


@dataclass
class RunRecord:
    config: str
    case: str
    ok: bool
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    n_segments: int = 0
    n_sources: int = 0
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "case": self.case,
            "ok": self.ok,
            "n_segments": self.n_segments,
            "n_sources": self.n_sources,
            "metrics": self.metrics,
            "timings": self.timings,
            "failure_reason": self.failure_reason,
        }


def run_case(
    cfg: SourceTraceConfig, data_dir: Path, out_dir: Path, iou_thr: float = 0.5
) -> tuple[EvalResult, dict, int, int]:
    """在一个数据集上跑一次并评估。返回 (评估结果, 各阶段耗时, 片段数, 入索引素材数)。"""
    gt = GroundTruth.load(data_dir / "ground_truth.json")
    res = run_trace(
        query=data_dir / "Final.mp4",
        sources=data_dir / "Sources",
        output=out_dir,
        cfg=cfg,
        export_clips=False,
        verify=False,
    )
    preds = [PredSegment.from_dict(s.to_dict()) for s in res.segments]
    ev = evaluate(gt, preds, iou_threshold=iou_thr)
    n_sources = int(res.result.get("meta", {}).get("n_sources_indexed", 0))
    return ev, res.timings, len(res.segments), n_sources


def benchmark(
    data_root: Path,
    out_root: Path,
    config_keys: list[str],
    cases: list[str],
    model: str = "isc21",
    iou_thr: float = 0.5,
) -> list[RunRecord]:
    log = get_logger()
    all_cfgs = make_configs(model)
    records: list[RunRecord] = []
    for key in config_keys:
        bc = all_cfgs.get(key)
        if bc is None:
            log.warning("未知的 benchmark 组合：%s，已跳过", key)
            continue
        for case in cases:
            data_dir = data_root / case
            if not (data_dir / "ground_truth.json").exists():
                log.warning("测试数据缺失，跳过：%s", data_dir)
                continue
            log.info("=" * 70)
            log.info("Benchmark %s（%s） × 用例 %s", key, bc.title, case)
            cfg = bc.build()  # type: ignore[operator]
            if key == "F" and not _transvcl_available():
                reason = "TransVCL 不可用（未安装可选依赖或缺少权重），已跳过该组合"
                log.warning("%s", reason)
                records.append(RunRecord(key, case, False, failure_reason=reason))
                continue
            t0 = time.perf_counter()
            try:
                ev, timings, n, n_src = run_case(cfg, data_dir, out_root / "runs" / f"{key}_{case}", iou_thr)
                rec = RunRecord(key, case, True, ev.to_dict(), timings, n, n_src)
                log.info("  " + "  ".join(ev.summary_lines()[:4]))
            except Exception as exc:
                rec = RunRecord(key, case, False, failure_reason=f"{type(exc).__name__}: {exc}")
                log.error("组合 %s / 用例 %s 运行失败：%s", key, case, exc)
            rec.timings.setdefault("wall", round(time.perf_counter() - t0, 3))
            records.append(rec)
    return records


def _transvcl_available() -> bool:
    try:
        from ..refinement.transvcl import is_available

        return is_available()
    except Exception:
        return False


def cache_probe(data_root: Path, case: str, out_root: Path, model: str = "isc21", config_key: str = "D") -> dict:
    """实测「首次运行」与「命中缓存的第二次运行」耗时。

    第一次运行前会清空特征缓存目录，因此得到的是真正的冷启动耗时。
    """
    log = get_logger()
    import shutil

    cfg = make_configs(model)[config_key].build()  # type: ignore[operator]
    data_dir = data_root / case
    out_dir = out_root / "runs" / f"cacheprobe_{case}"
    cache_dir = cfg.cache_dir(out_dir.parent)
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    log.info("缓存冷启动测试：已清空 %s", cache_dir)
    _, t_cold, _, _ = run_case(cfg, data_dir, out_dir)
    _, t_warm, _, _ = run_case(cfg, data_dir, out_dir)
    probe = {
        "case": case,
        "config": config_key,
        "cold_total": t_cold.get("total"),
        "cold_feature_sources": t_cold.get("feature_sources"),
        "warm_total": t_warm.get("total"),
        "warm_feature_sources": t_warm.get("feature_sources"),
    }
    log.info("缓存对比：首次 %.2fs（素材特征 %.2fs）→ 第二次 %.2fs（素材特征 %.2fs）",
             probe["cold_total"] or 0.0, probe["cold_feature_sources"] or 0.0,
             probe["warm_total"] or 0.0, probe["warm_feature_sources"] or 0.0)
    return probe


# ---------------------------------------------------------------- 报告


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return str(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f < 0:
        return "-"
    return f"{f:.{nd}f}"


def write_reports(records: list[RunRecord], out_root: Path, env: dict, probe: dict | None = None) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    payload = {"env": env, "runs": [r.to_dict() for r in records]}
    if probe:
        payload["cache_probe"] = probe
    (out_root / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["results"] = out_root / "results.json"

    ok = [r for r in records if r.ok]

    # ---- BENCHMARK_REPORT ----
    lines = ["# BENCHMARK_REPORT", "", "## 运行环境", ""]
    for k, v in env.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 各算法组合 × 用例 指标", ""]
    for r in records:
        if not r.ok:
            lines.append(f"### {r.config} / {r.case}：失败")
            lines.append(f"- FAILURE_REASON: {r.failure_reason}")
            lines.append("")
            continue
        m = r.metrics
        lines += [
            f"### {r.config} / {r.case}",
            f"- source_recall: {_fmt(m.get('source_recall'))}",
            f"- segment_recall(IoU>={_fmt(m.get('iou_threshold'), 2)}): {_fmt(m.get('segment_recall'))}",
            f"- mean_start_error: {_fmt(m.get('mean_start_error'))} s",
            f"- mean_end_error: {_fmt(m.get('mean_end_error'))} s",
            f"- median_iou(source): {_fmt(m.get('median_iou'))}",
            f"- median_iou_2d: {_fmt(m.get('median_iou_2d'))}",
            f"- false_positive_rate: {_fmt(m.get('false_positive_rate'))}",
            f"- unknown_accuracy: {_fmt(m.get('unknown_accuracy'))}",
            f"- 片段数: {r.n_segments}，耗时: {_fmt(r.timings.get('total'), 2)} s",
            "",
        ]

    # 汇总：按组合平均
    lines += ["## 组合汇总（跨用例平均）", ""]
    by_cfg: dict[str, list[RunRecord]] = {}
    for r in ok:
        by_cfg.setdefault(r.config, []).append(r)
    ranking = []
    for key, rs in by_cfg.items():
        def avg(field_: str) -> float:
            vals = [rr.metrics.get(field_, -1.0) for rr in rs]
            vals = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
            return sum(vals) / len(vals) if vals else -1.0

        item = {
            "config": key,
            "source_recall": avg("source_recall"),
            "segment_recall": avg("segment_recall"),
            "mean_start_error": avg("mean_start_error"),
            "mean_end_error": avg("mean_end_error"),
            "median_iou": avg("median_iou"),
            "fpr": avg("false_positive_rate"),
            "time": sum(rr.timings.get("total", 0.0) for rr in rs) / len(rs),
            "vram": max(rr.timings.get("peak_vram_mb", 0.0) for rr in rs),
        }
        ranking.append(item)
        lines += [
            f"- **{key}**：source_recall={_fmt(item['source_recall'])}  "
            f"segment_recall={_fmt(item['segment_recall'])}  "
            f"start_err={_fmt(item['mean_start_error'])}s  end_err={_fmt(item['mean_end_error'])}s  "
            f"IoU={_fmt(item['median_iou'])}  FPR={_fmt(item['fpr'])}  "
            f"耗时={_fmt(item['time'], 2)}s  峰值显存={_fmt(item['vram'], 1)}MB"
        ]

    if ranking:
        best_acc = max(ranking, key=lambda x: (x["segment_recall"], x["source_recall"], -x["mean_start_error"]))
        fastest = min(ranking, key=lambda x: x["time"])
        best_loc = min(
            [r for r in ranking if r["mean_start_error"] >= 0] or ranking,
            key=lambda x: (x["mean_start_error"] + x["mean_end_error"]) / 2,
        )
        lowest_fp = min(ranking, key=lambda x: x["fpr"])
        lines += [
            "",
            "## 结论",
            "",
            f"- 综合最优：**{best_acc['config']}**（segment_recall={_fmt(best_acc['segment_recall'])}）",
            f"- 最快：**{fastest['config']}**（平均 {_fmt(fastest['time'], 2)}s，"
            f"segment_recall={_fmt(fastest['segment_recall'])}）",
            f"- 时间定位最准：**{best_loc['config']}**（起止平均误差 "
            f"{_fmt((best_loc['mean_start_error'] + best_loc['mean_end_error']) / 2)}s，"
            f"segment_recall={_fmt(best_loc['segment_recall'])}）",
            f"- 误匹配最低：**{lowest_fp['config']}**（FPR={_fmt(lowest_fp['fpr'])}，"
            f"segment_recall={_fmt(lowest_fp['segment_recall'])}）",
            f"- 峰值显存最高：**{max(ranking, key=lambda x: x['vram'])['config']}**",
            "",
            "> 注意：start_err / end_err / IoU 只在「已正确匹配到来源」的片段上统计，"
            "因此 recall 低的组合这些数值会偏乐观，必须与 segment_recall 一起看。",
        ]
    (out_root / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["benchmark"] = out_root / "BENCHMARK_REPORT.md"

    # ---- PERFORMANCE_REPORT ----
    plines = ["# PERFORMANCE_REPORT", "", "## 环境", ""]
    for k, v in env.items():
        plines.append(f"- {k}: {v}")
    plines += ["", "## 各阶段耗时（秒）", ""]
    for r in ok:
        t = r.timings
        plines.append(
            f"- {r.config}/{r.case}: total={_fmt(t.get('total'), 2)}  "
            f"特征(素材)={_fmt(t.get('feature_sources'), 2)}  特征(成片)={_fmt(t.get('feature_query'), 2)}  "
            f"镜头={_fmt(t.get('shot_detect'), 2)}  检索={_fmt(t.get('retrieval'), 3)}  "
            f"粗对齐={_fmt(t.get('align_coarse'), 3)}  全片重扫={_fmt(t.get('align_rescan'), 2)}  "
            f"精定位={_fmt(t.get('refine_fine'), 2)}  "
            f"困难样本={_fmt(t.get('refine_hard'), 2)}  导出={_fmt(t.get('export'), 2)}  "
            f"峰值显存={_fmt(t.get('peak_vram_mb'), 1)}MB"
        )
    if probe:
        plines += [
            "",
            "## 缓存冷启动 / 热启动对比（实测）",
            "",
            f"- 用例 {probe.get('case')}，组合 {probe.get('config')}",
            f"- 首次运行（清空缓存）：total={_fmt(probe.get('cold_total'), 2)}s，"
            f"其中素材特征提取={_fmt(probe.get('cold_feature_sources'), 2)}s",
            f"- 第二次运行（命中缓存）：total={_fmt(probe.get('warm_total'), 2)}s，"
            f"其中素材特征提取={_fmt(probe.get('warm_feature_sources'), 2)}s",
        ]
    (out_root / "PERFORMANCE_REPORT.md").write_text("\n".join(plines) + "\n", encoding="utf-8")
    paths["performance"] = out_root / "PERFORMANCE_REPORT.md"

    # ---- ROBUSTNESS_REPORT ----
    rlines = ["# ROBUSTNESS_REPORT", "",
              "按用例查看鲁棒性（robust 用例逐项覆盖 纯裁剪/缩放/重编码/调色/字幕/水印/画面裁切/变速/两次加工）", ""]
    for r in ok:
        m = r.metrics
        rlines += [
            f"## {r.config} / {r.case}",
            f"- source_recall={_fmt(m.get('source_recall'))}  segment_recall={_fmt(m.get('segment_recall'))}"
            f"  start_err={_fmt(m.get('mean_start_error'))}s  IoU={_fmt(m.get('median_iou'))}",
            "",
            "| GT | 来源正确 | 起点误差(s) | 终点误差(s) | IoU | 状态 | 备注 |",
            "| -- | -- | -- | -- | -- | -- | -- |",
        ]
        for d in m.get("details", []):
            rlines.append(
                f"| {d['gt_id']} | {'是' if d['source_correct'] else '否'} | "
                f"{_fmt(d.get('start_error'), 3)} | {_fmt(d.get('end_error'), 3)} | "
                f"{_fmt(d.get('iou'), 3)} | {d.get('status')} | {d.get('note', '')} |"
            )
        rlines.append("")
    (out_root / "ROBUSTNESS_REPORT.md").write_text("\n".join(rlines) + "\n", encoding="utf-8")
    paths["robustness"] = out_root / "ROBUSTNESS_REPORT.md"

    return paths


def collect_env() -> dict:
    env = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "cpu": platform.processor(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_mb"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024 / 1024:.0f}"
        env["cuda_version"] = str(torch.version.cuda)
    except Exception as exc:
        env["torch"] = f"不可用（{exc}）"
    try:
        from ..video.ffmpeg_tools import check_available

        info = check_available()
        env["ffmpeg"] = info["ffmpeg"]
    except Exception as exc:
        env["ffmpeg"] = f"不可用（{exc}）"
    return env


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="自动 benchmark 与报告生成")
    p.add_argument("--data", default="testdata", help="测试数据根目录（其下每个子目录为一个用例）")
    p.add_argument("--out", default="reports", help="报告输出目录")
    p.add_argument("--configs", default="A,B,C,C-dp,C-dtw,C-hv,D,E,F", help="算法组合，逗号分隔")
    p.add_argument("--cases", default="basic,multi,robust,distractor,unknown", help="用例，逗号分隔")
    p.add_argument("--model", default="isc21", choices=["isc21", "colorhist"], help="描述子模型")
    p.add_argument("--iou", type=float, default=0.5, help="segment_recall 的 IoU 阈值")
    p.add_argument(
        "--cache-probe",
        default="",
        help="额外实测该用例的「清空缓存首次运行 vs 命中缓存第二次运行」耗时，例如 --cache-probe basic",
    )
    args = p.parse_args(argv)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=out_root / "benchmark.log")

    env = collect_env()
    env["model"] = args.model
    config_keys = [c.strip() for c in args.configs.split(",") if c.strip()]
    records = benchmark(
        Path(args.data),
        out_root,
        config_keys,
        [c.strip() for c in args.cases.split(",") if c.strip()],
        model=args.model,
        iou_thr=args.iou,
    )
    probe = None
    if args.cache_probe:
        probe = cache_probe(Path(args.data), args.cache_probe, out_root, model=args.model)
    if not records:
        # 只跑缓存实测时，合并进已有的 results.json 并重新出报告
        old = out_root / "results.json"
        if old.exists() and probe:
            data = json.loads(old.read_text(encoding="utf-8"))
            data["cache_probe"] = probe
            old.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            get_logger().info("已把缓存实测结果合并进 %s", old)
            return 0
    paths = write_reports(records, out_root, env, probe)
    log = get_logger()
    for k, v in paths.items():
        log.info("报告已生成 %s -> %s", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
