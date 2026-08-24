"""帧抽取（基于 FFmpeg rawvideo 管道）。

设计要点：
* 时间戳来源于 FFmpeg ``showinfo`` 滤镜输出的 ``pts_time``，而不是 ``frame_index/fps``，
  因此对 VFR 视频同样正确；解析失败时才退化为 ``i/fps`` 估算。
* 支持区间抽帧：前置 ``-ss`` 粗跳 + 后置 ``-ss`` 精确定位，兼顾速度与精度。
* stdout 读取 rawvideo，stderr 由独立线程读取，避免管道死锁。
"""

from __future__ import annotations

import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.log import get_logger
from .ffmpeg_tools import ffmpeg_path
from .probe import VideoInfo

_PTS_RE = re.compile(r"pts_time:\s*([0-9]+\.?[0-9]*)")
_PRE_SEEK_PAD = 3.0  # 前置 -ss 之前预留的安全时间


class FrameReadError(RuntimeError):
    pass


@dataclass
class FrameBatch:
    """一次抽帧结果。"""

    frames: np.ndarray  # (N, H, W, C) uint8，C=3(RGB) 或 1(GRAY)
    timestamps: np.ndarray  # (N,) float64，单位秒，基于原视频时间轴
    fps: float
    width: int
    height: int
    timestamp_source: str  # "pts" / "estimated"

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def extract_frames(
    info: VideoInfo,
    fps: float,
    size: tuple[int, int] | int,
    start: float | None = None,
    end: float | None = None,
    gray: bool = False,
) -> FrameBatch:
    """按指定帧率抽帧。

    参数
    ----
    info : 已探测的视频信息
    fps  : 目标采样帧率
    size : (width, height) 或短边整数（正方形 resize）
    start/end : 抽帧区间（秒，基于原视频时间轴），None 表示整段
    gray : 是否输出单通道灰度（用于镜头检测，省内存）
    """
    if isinstance(size, int):
        out_w = out_h = int(size)
    else:
        out_w, out_h = int(size[0]), int(size[1])

    duration = info.duration
    seg_start = 0.0 if start is None else max(0.0, float(start))
    seg_end = duration if end is None else min(duration, float(end))
    if seg_end - seg_start <= 1e-6:
        empty_c = 1 if gray else 3
        return FrameBatch(
            frames=np.zeros((0, out_h, out_w, empty_c), dtype=np.uint8),
            timestamps=np.zeros((0,), dtype=np.float64),
            fps=fps,
            width=out_w,
            height=out_h,
            timestamp_source="pts",
        )

    pre_seek = max(0.0, seg_start - _PRE_SEEK_PAD)
    seg_dur = seg_end - seg_start

    pix_fmt = "gray" if gray else "rgb24"
    channels = 1 if gray else 3
    # 关键：区间裁剪必须放在滤镜链里（trim），这样 showinfo 打印的帧与写出的帧严格一一对应。
    # 若改用输出侧 -ss，showinfo 会把被丢弃的帧也打印出来，导致时间戳与帧错位。
    trim_end = seg_end + 1.5 / max(fps, 1e-6)
    vf = (
        f"trim=start={seg_start:.6f}:end={trim_end:.6f},"
        f"fps={fps:.6f},scale={out_w}:{out_h}:flags=bicubic,showinfo"
    )

    args = [ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "info"]
    if pre_seek > 0:
        args += ["-ss", f"{pre_seek:.6f}"]
    args += ["-i", str(info.path)]
    args += [
        # -copyts 保留原始时间轴，使 showinfo 的 pts_time 直接是原视频时间
        "-copyts",
        "-an", "-sn", "-dn",
        "-fps_mode", "passthrough",
        "-vf", vf,
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "pipe:1",
    ]

    frame_bytes = out_w * out_h * channels
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes * 4)

    stderr_chunks: list[bytes] = []

    def _drain() -> None:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, b""):
            stderr_chunks.append(line)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    buffers: list[bytes] = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(frame_bytes)
        if not chunk:
            break
        if len(chunk) < frame_bytes:  # 补齐最后一帧
            remain = frame_bytes - len(chunk)
            tail = proc.stdout.read(remain)
            chunk += tail or b""
            if len(chunk) < frame_bytes:
                break
        buffers.append(chunk)
    proc.stdout.close()
    proc.wait()
    t.join(timeout=5)

    stderr_text = b"".join(stderr_chunks).decode("utf-8", "ignore")
    if not buffers:
        raise FrameReadError(
            f"抽帧失败（未获得任何帧）：{Path(info.path).name} "
            f"区间[{seg_start:.3f},{seg_end:.3f}]\n{stderr_text[-1500:]}"
        )

    n = len(buffers)
    frames = np.frombuffer(b"".join(buffers), dtype=np.uint8).reshape(n, out_h, out_w, channels)

    pts = [float(m) for m in _PTS_RE.findall(stderr_text)]
    plausible = (
        len(pts) >= n
        and pts[0] >= seg_start - 1.0
        and pts[0] <= seg_start + 2.0
        and all(b > a for a, b in zip(pts[:n], pts[1:n]))
    )
    if plausible:
        timestamps = np.asarray(pts[:n], dtype=np.float64)
        ts_source = "pts"
    else:
        get_logger().debug(
            "showinfo 时间戳不可用(%d 条, first=%s)，退化为 i/fps 估算：%s",
            len(pts), pts[0] if pts else None, Path(info.path).name,
        )
        timestamps = seg_start + np.arange(n, dtype=np.float64) / float(fps)
        ts_source = "estimated"

    # 裁掉超出区间的尾帧（-frames:v 多取了余量）
    keep = timestamps <= seg_end + 1e-6
    if not keep.all():
        frames = frames[keep]
        timestamps = timestamps[keep]

    return FrameBatch(
        frames=np.ascontiguousarray(frames),
        timestamps=timestamps,
        fps=fps,
        width=out_w,
        height=out_h,
        timestamp_source=ts_source,
    )


def iter_frames(
    info: VideoInfo,
    fps: float,
    size: tuple[int, int] | int,
    chunk_sec: float = 60.0,
    gray: bool = False,
):
    """分块抽帧，避免长视频一次性占满内存。逐块 yield FrameBatch。"""
    t = 0.0
    while t < info.duration - 1e-6:
        end = min(info.duration, t + chunk_sec)
        batch = extract_frames(info, fps=fps, size=size, start=t, end=end, gray=gray)
        if len(batch):
            yield batch
        t = end
