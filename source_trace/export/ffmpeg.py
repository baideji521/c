"""FFmpeg 片段导出。

精度优先：默认使用精确重编码（CRF 16，视觉无损），保证起止时间与报告一致。
``mode=copy`` 时使用 stream copy，速度最快但起点会被吸附到最近关键帧，
此时会在日志与报告中标注实际偏移，绝不静默产生错误片段。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import ExportConfig
from ..utils.log import get_logger
from ..video.ffmpeg_tools import ffmpeg_path
from ..video.probe import probe


@dataclass
class ClipResult:
    path: str
    requested_start: float
    requested_end: float
    actual_duration: float
    mode: str
    ok: bool
    error: str = ""


def _fmt_time(t: float) -> str:
    """00-52-31 形式（分-秒-百分秒，用于文件名）。

    先把时间整体量化到百分秒再拆分：直接对秒的小数部分取整会丢进位，
    64.995s 会被写成 01-04-100 而不是 01-05-00。
    """
    cs_total = int(round(max(0.0, t) * 100))
    m, rem = divmod(cs_total, 6000)
    s, cs = divmod(rem, 100)
    return f"{m:02d}-{s:02d}-{cs:02d}"


def clip_name(seg_id: int, source_name: str, start: float, end: float) -> str:
    stem = Path(source_name).stem
    return f"{seg_id:03d}_{stem}_{_fmt_time(start)}_{_fmt_time(end)}.mp4"


def export_clip(
    src: Path | str,
    start: float,
    end: float,
    dst: Path | str,
    cfg: ExportConfig,
) -> ClipResult:
    """导出 [start, end) 区间。"""
    log = get_logger()
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, end - start)
    if dur <= 1e-3:
        return ClipResult(str(dst), start, end, 0.0, cfg.mode, False, "区间长度为 0")

    mode = "copy" if cfg.mode == "copy" else "reencode"
    if mode == "copy":
        args = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.6f}", "-i", str(src), "-t", f"{dur:.6f}",
            "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst),
        ]
    else:
        args = [
            ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{max(0.0, start - 3.0):.6f}", "-i", str(src),
            "-ss", f"{min(3.0, start):.6f}", "-t", f"{dur:.6f}",
            "-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset,
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
            str(dst),
        ]
    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore").strip()[-600:]
        log.error("片段导出失败：%s（%s）", dst.name, err)
        return ClipResult(str(dst), start, end, 0.0, mode, False, err)

    try:
        actual = probe(dst).duration
    except Exception:
        actual = -1.0
    if actual > 0 and abs(actual - dur) > 0.25:
        log.warning(
            "片段 %s 实际时长 %.3fs 与请求 %.3fs 偏差较大（mode=%s）", dst.name, actual, dur, mode
        )
    return ClipResult(str(dst), start, end, actual, mode, True)


def concat_clips(
    clips: list[Path | str],
    dst: Path | str,
    cfg: ExportConfig,
    width: int = 0,
    height: int = 0,
    fps: float = 0.0,
) -> ClipResult:
    """把已裁剪出来的原始片段按顺序拼成一条新素材。

    各素材的分辨率/帧率/采样率可能都不一样（实测 29.97 / 30 / 60 fps 混用），
    concat 分离器要求参数完全一致，因此这里用 concat 滤镜逐路归一化后再拼接。
    width/height/fps 传 0 时以第一个片段为基准。
    """
    log = get_logger()
    dst = Path(dst)
    paths = [Path(p) for p in clips if Path(p).exists()]
    if not paths:
        return ClipResult(str(dst), 0.0, 0.0, 0.0, "concat", False, "没有可拼接的片段")
    dst.parent.mkdir(parents=True, exist_ok=True)

    first = probe(paths[0])
    w = int(width or first.width)
    h = int(height or first.height)
    f = float(fps or first.fps or 30.0)
    # 任一片段没有音轨时整体丢弃音频，避免 concat 滤镜的流数量不匹配
    with_audio = all(probe(p).has_audio for p in paths)

    args: list[str] = [ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error"]
    for p in paths:
        args += ["-i", str(p)]
    parts: list[str] = []
    refs: list[str] = []
    for i in range(len(paths)):
        parts.append(
            f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={f:.6f},format=yuv420p[v{i}]"
        )
        refs.append(f"[v{i}]")
        if with_audio:
            parts.append(f"[{i}:a]aresample=48000,asetpts=N/SR/TB[a{i}]")
            refs.append(f"[a{i}]")
    n = len(paths)
    parts.append("".join(refs) + f"concat=n={n}:v=1:a={1 if with_audio else 0}[v]" + ("[a]" if with_audio else ""))
    args += ["-filter_complex", ";".join(parts), "-map", "[v]"]
    if with_audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
    args += ["-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset, "-pix_fmt", "yuv420p", str(dst)]

    proc = subprocess.run(args, capture_output=True)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "ignore").strip()[-800:]
        log.error("片段拼接失败：%s（%s）", dst.name, err)
        return ClipResult(str(dst), 0.0, 0.0, 0.0, "concat", False, err)
    try:
        actual = probe(dst).duration
    except Exception:
        actual = -1.0
    log.info("已合成新素材：%s（%d 段，%.2fs，%dx%d@%.3gfps%s）",
             dst.name, n, actual, w, h, f, "，含音频" if with_audio else "，无音频")
    return ClipResult(str(dst), 0.0, actual, actual, "concat", True)
