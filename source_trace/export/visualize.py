"""可视化验证：contact sheet 与左右对照视频。

人工只需扫一眼 ``verification/segment_XXX.jpg`` 就能判断反溯是否正确：
上排为成片片段抽帧，下排为对应原始素材片段抽帧，时间戳直接标在图上。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..config import ExportConfig
from ..temporal.mapping import TimeSegment
from ..utils.log import get_logger
from ..video.ffmpeg_tools import ffmpeg_path
from ..video.probe import VideoInfo, probe
from ..video.reader import extract_frames


def _sample_grid(info: VideoInfo, start: float, end: float, n: int, width: int) -> list[np.ndarray]:
    """在 [start, end] 均匀取 n 帧（BGR）。"""
    out: list[np.ndarray] = []
    if end <= start:
        return out
    times = np.linspace(start, max(start, end - 1e-3), n)
    h = int(round(width * info.height / max(info.width, 1) / 2) * 2)
    for t in times:
        try:
            fb = extract_frames(info, fps=25.0, size=(width, h), start=float(t), end=float(t) + 0.08)
        except Exception:
            continue
        if len(fb) == 0:
            continue
        rgb = fb.frames[0]
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(img, (0, h - 22), (width, h), (0, 0, 0), -1)
        cv2.putText(img, f"{t:8.2f}s", (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        out.append(img)
    return out


def _row(images: list[np.ndarray], label: str, width: int) -> np.ndarray | None:
    if not images:
        return None
    h = max(i.shape[0] for i in images)
    canvas = np.zeros((h + 26, width * len(images), 3), dtype=np.uint8)
    for k, img in enumerate(images):
        canvas[26 : 26 + img.shape[0], k * width : k * width + img.shape[1]] = img
    cv2.putText(canvas, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def contact_sheet(
    query_info: VideoInfo,
    source_info: VideoInfo | None,
    seg: TimeSegment,
    out_path: Path,
    cfg: ExportConfig,
) -> Path | None:
    """生成成片/原素材上下对照的 contact sheet。"""
    log = get_logger()
    w = cfg.sheet_thumb_width
    n = max(2, cfg.sheet_columns)
    q_imgs = _sample_grid(query_info, seg.query_start, seg.query_end, n, w)
    rows = []
    top = _row(q_imgs, f"FINAL  {seg.query_start:.2f}-{seg.query_end:.2f}s", w)
    if top is not None:
        rows.append(top)
    if source_info is not None and seg.source_start is not None and seg.source_end is not None:
        s_imgs = _sample_grid(source_info, seg.source_start, seg.source_end, n, w)
        bottom = _row(
            s_imgs,
            f"{seg.source}  {seg.source_start:.2f}-{seg.source_end:.2f}s  "
            f"conf={seg.confidence:.3f} {seg.status}",
            w,
        )
        if bottom is not None:
            rows.append(bottom)
    if not rows:
        return None
    width = max(r.shape[1] for r in rows)
    canvas = np.zeros((sum(r.shape[0] for r in rows) + 6 * len(rows), width, 3), dtype=np.uint8)
    y = 0
    for r in rows:
        canvas[y : y + r.shape[0], : r.shape[1]] = r
        y += r.shape[0] + 6
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), canvas)
    if not ok:
        log.warning("contact sheet 写入失败：%s", out_path)
        return None
    return out_path


def compare_video(
    query_path: Path | str,
    source_path: Path | str,
    seg: TimeSegment,
    out_path: Path,
    height: int = 360,
) -> Path | None:
    """生成左（成片）右（原素材）并排对照视频。"""
    log = get_logger()
    if seg.source_start is None or seg.source_end is None:
        return None
    q_dur = seg.query_end - seg.query_start
    s_dur = seg.source_end - seg.source_start
    if q_dur <= 0.05 or s_dur <= 0.05:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{seg.query_start:.6f}", "-t", f"{q_dur:.6f}", "-i", str(query_path),
        "-ss", f"{seg.source_start:.6f}", "-t", f"{s_dur:.6f}", "-i", str(source_path),
        "-filter_complex",
        (
            f"[0:v]scale=-2:{height},setsar=1,drawtext=text='FINAL':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.5"
            f"[l];"
            f"[1:v]scale=-2:{height},setsar=1,drawtext=text='SOURCE':x=10:y=10:fontcolor=white:box=1:boxcolor=black@0.5"
            f"[r];[l][r]hstack=inputs=2"
        ),
        "-an", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        # drawtext 依赖字体，缺失时退化为无文字版本
        args_nodraw = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{seg.query_start:.6f}", "-t", f"{q_dur:.6f}", "-i", str(query_path),
            "-ss", f"{seg.source_start:.6f}", "-t", f"{s_dur:.6f}", "-i", str(source_path),
            "-filter_complex",
            f"[0:v]scale=-2:{height},setsar=1[l];[1:v]scale=-2:{height},setsar=1[r];[l][r]hstack=inputs=2",
            "-an", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc2 = subprocess.run(args_nodraw, capture_output=True)
        if proc2.returncode != 0:
            log.warning("对照视频生成失败：%s", proc2.stderr.decode("utf-8", "ignore")[-400:])
            return None
    return out_path


def make_verification(
    query_path: Path | str,
    sources_dir: Path | str,
    segments: list[TimeSegment],
    out_dir: Path | str,
    cfg: ExportConfig,
) -> list[dict]:
    """批量生成验证材料。"""
    log = get_logger()
    out_dir = Path(out_dir)
    q_info = probe(query_path)
    cache: dict[str, VideoInfo] = {}
    results = []
    for seg in segments:
        item: dict = {"segment_id": seg.id}
        s_info = None
        if seg.source:
            if seg.source not in cache:
                try:
                    cache[seg.source] = probe(Path(sources_dir) / seg.source)
                except Exception as exc:
                    log.warning("验证图生成跳过（无法读取 %s）：%s", seg.source, exc)
                    cache[seg.source] = None  # type: ignore[assignment]
            s_info = cache.get(seg.source)
        if cfg.contact_sheet:
            p = contact_sheet(q_info, s_info, seg, out_dir / f"segment_{seg.id:03d}.jpg", cfg)
            item["sheet"] = str(p) if p else None
        if cfg.compare_video and s_info is not None:
            p = compare_video(
                query_path, s_info.path, seg, out_dir / f"segment_{seg.id:03d}_compare.mp4"
            )
            item["compare"] = str(p) if p else None
        results.append(item)
    log.info("验证材料已生成：%s（%d 项）", out_dir, len(results))
    return results
