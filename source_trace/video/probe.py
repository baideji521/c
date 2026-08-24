"""视频元数据探测（基于 ffprobe）。

所有时间统一为秒（float64）。不依赖 frame index 作为最终时间来源。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.log import get_logger
from .ffmpeg_tools import ffprobe_path, run


class VideoProbeError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float  # avg_frame_rate
    r_fps: float  # r_frame_rate（容器标称）
    duration: float
    frame_count: int
    codec: str
    pix_fmt: str
    start_time: float
    has_audio: bool
    is_vfr: bool
    rotation: int = 0
    nb_read_frames: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def orientation(self) -> str:
        if self.height > self.width:
            return "portrait"
        if self.height < self.width:
            return "landscape"
        return "square"

    def describe(self) -> str:
        return (
            f"{Path(self.path).name}  {self.width}x{self.height}({self.orientation})  "
            f"fps={self.fps:.3f}{'(VFR)' if self.is_vfr else ''}  时长={self.duration:.3f}s  "
            f"帧数={self.frame_count}  codec={self.codec}  音频={'有' if self.has_audio else '无'}"
        )


def _parse_rate(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe(path: Path | str, count_frames: bool = False) -> VideoInfo:
    """读取视频元数据。count_frames=True 时精确统计帧数（较慢）。"""
    path = Path(path)
    if not path.exists():
        raise VideoProbeError(f"文件不存在：{path}")

    args = [
        ffprobe_path(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
    ]
    if count_frames:
        args.append("-count_frames")
    args.append(str(path))

    proc = run(args)
    if proc.returncode != 0:
        raise VideoProbeError(
            f"ffprobe 解析失败：{path.name}\n{(proc.stderr or b'').decode('utf-8', 'ignore').strip()}"
        )
    try:
        data = json.loads((proc.stdout or b"").decode("utf-8", "ignore"))
    except json.JSONDecodeError as exc:
        raise VideoProbeError(f"ffprobe 输出无法解析：{path.name}") from exc

    streams = data.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    if not vstreams:
        raise VideoProbeError(f"未找到视频流：{path.name}")
    v = vstreams[0]
    fmt = data.get("format", {})

    avg_fps = _parse_rate(v.get("avg_frame_rate"))
    r_fps = _parse_rate(v.get("r_frame_rate"))

    duration = 0.0
    for src in (v.get("duration"), fmt.get("duration")):
        try:
            duration = float(src)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    frame_count = 0
    for key in ("nb_read_frames", "nb_frames"):
        try:
            frame_count = int(v.get(key))
            if frame_count > 0:
                break
        except (TypeError, ValueError):
            continue
    if frame_count <= 0 and duration > 0 and avg_fps > 0:
        frame_count = int(round(duration * avg_fps))

    if duration <= 0 and frame_count > 0 and avg_fps > 0:
        duration = frame_count / avg_fps
    if duration <= 0:
        raise VideoProbeError(f"无法确定视频时长：{path.name}")

    try:
        start_time = float(v.get("start_time", fmt.get("start_time", 0.0)))
    except (TypeError, ValueError):
        start_time = 0.0

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                rotation = 0
    width, height = int(v.get("width", 0)), int(v.get("height", 0))
    if rotation in (90, 270):
        width, height = height, width

    # VFR 判定：avg_frame_rate 与 r_frame_rate 显著不一致
    is_vfr = bool(avg_fps and r_fps and abs(avg_fps - r_fps) / max(avg_fps, r_fps) > 0.02)

    info = VideoInfo(
        path=str(path.resolve()),
        width=width,
        height=height,
        fps=avg_fps or r_fps,
        r_fps=r_fps or avg_fps,
        duration=duration,
        frame_count=frame_count,
        codec=str(v.get("codec_name", "unknown")),
        pix_fmt=str(v.get("pix_fmt", "unknown")),
        start_time=start_time,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        is_vfr=is_vfr,
        rotation=rotation,
        nb_read_frames=int(v["nb_read_frames"]) if count_frames and v.get("nb_read_frames") else None,
        extra={"format_name": fmt.get("format_name", ""), "bit_rate": fmt.get("bit_rate", "")},
    )
    get_logger().debug("探测：%s", info.describe())
    return info


def scan_videos(directory: Path | str, exts: tuple[str, ...]) -> list[Path]:
    """扫描目录下所有支持的视频文件（按文件名排序，递归）。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise VideoProbeError(f"素材目录不存在：{directory}")
    lower = tuple(e.lower() for e in exts)
    files = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in lower]
    return files
