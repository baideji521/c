"""FFmpeg / FFprobe 可执行文件定位与调用封装。

查找顺序：
1. 环境变量 SOURCE_TRACE_FFMPEG / SOURCE_TRACE_FFPROBE
2. 项目内 tools/ffmpeg/
3. 系统 PATH
4. imageio-ffmpeg 自带的二进制（仅 ffmpeg，无 ffprobe）
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOCAL_BIN = _PROJECT_ROOT / "tools" / "ffmpeg"


class FFmpegNotFound(RuntimeError):
    pass


def _candidates(name: str, env_key: str) -> list[Path]:
    out: list[Path] = []
    env = os.environ.get(env_key)
    if env:
        out.append(Path(env))
    exe = f"{name}.exe" if os.name == "nt" else name
    out.append(_LOCAL_BIN / exe)
    found = shutil.which(name)
    if found:
        out.append(Path(found))
    return out


@lru_cache(maxsize=None)
def ffmpeg_path() -> str:
    for c in _candidates("ffmpeg", "SOURCE_TRACE_FFMPEG"):
        if c.exists():
            return str(c)
    try:  # imageio-ffmpeg 兜底
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise FFmpegNotFound(
        "未找到 ffmpeg。请将 ffmpeg.exe / ffprobe.exe 放入 tools/ffmpeg/，"
        "或加入系统 PATH，或设置环境变量 SOURCE_TRACE_FFMPEG。"
    )


@lru_cache(maxsize=None)
def ffprobe_path() -> str:
    for c in _candidates("ffprobe", "SOURCE_TRACE_FFPROBE"):
        if c.exists():
            return str(c)
    raise FFmpegNotFound(
        "未找到 ffprobe。请将 ffprobe.exe 放入 tools/ffmpeg/ 或加入系统 PATH。"
    )


def run(args: list[str], capture: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    """执行外部命令（参数以列表传入，避免 shell 注入）。"""
    return subprocess.run(
        args,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        timeout=timeout,
        check=False,
    )


def check_available() -> dict[str, str]:
    """返回 ffmpeg/ffprobe 版本信息，供启动时打印。"""
    info: dict[str, str] = {}
    for name, getter in (("ffmpeg", ffmpeg_path), ("ffprobe", ffprobe_path)):
        exe = getter()
        proc = run([exe, "-version"])
        first = (proc.stdout or b"").decode("utf-8", "ignore").splitlines()
        info[name] = first[0] if first else "unknown"
        info[f"{name}_path"] = exe
    return info
