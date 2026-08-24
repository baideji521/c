"""视频读取与时间戳正确性测试。

关键断言：抽帧时间戳来自 pts，与画面中嵌入的帧序号一致（误差 <= 1 帧）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from source_trace.video.probe import probe, scan_videos
from source_trace.video.reader import extract_frames

FPS = 25.0


def _decode_frame_code(gray: np.ndarray, w: int, h: int) -> int:
    """解码合成素材右下角嵌入的帧序号二进制码。"""
    cell = max(6, w // 80)
    x0, y0 = w - cell * 18 - 8, h - cell - 8
    val = 0
    for b in range(18):
        patch = gray[y0 : y0 + cell, x0 + b * cell : x0 + (b + 1) * cell]
        if patch.mean() > 127:
            val |= 1 << b
    return val


def test_probe_metadata(source01: Path):
    info = probe(source01)
    assert info.width == 854 and info.height == 480
    assert info.fps == pytest.approx(FPS, abs=0.01)
    assert info.duration == pytest.approx(60.0, abs=0.1)
    assert info.frame_count == 1500
    assert info.codec == "h264"
    assert info.has_audio is False
    assert info.orientation == "landscape"
    assert info.is_vfr is False


def test_scan_videos(basic_dataset: Path):
    files = scan_videos(basic_dataset / "Sources", (".mp4", ".mov", ".mkv", ".webm"))
    assert [f.name for f in files] == ["Source01.mp4", "Source02.mp4", "Source03.mp4"]


@pytest.mark.parametrize("t", [0.0, 7.32, 17.5, 30.0, 52.0])
def test_timestamp_matches_embedded_frame_code(source01: Path, t: float):
    info = probe(source01)
    fb = extract_frames(info, fps=FPS, size=(854, 480), start=t, end=t + 0.05)
    assert len(fb) >= 1
    assert fb.timestamp_source == "pts"
    gray = fb.frames[0].mean(axis=2)
    idx = _decode_frame_code(gray, 854, 480)
    # 画面内的真实时间 与 报告的时间戳 误差不超过 1 帧
    assert abs(idx / FPS - fb.timestamps[0]) <= 1.0 / FPS + 1e-6


def test_sampling_rate_and_range(source01: Path):
    info = probe(source01)
    fb = extract_frames(info, fps=2.0, size=64, start=10.0, end=15.0)
    assert 9 <= len(fb) <= 12
    assert fb.timestamps.min() >= 10.0 - 1e-6
    assert fb.timestamps.max() <= 15.0 + 1e-6
    assert np.all(np.diff(fb.timestamps) > 0)
    assert fb.frames.shape[1:] == (64, 64, 3)


def test_gray_mode(source01: Path):
    info = probe(source01)
    fb = extract_frames(info, fps=1.0, size=32, start=0.0, end=3.0, gray=True)
    assert fb.frames.shape[1:] == (32, 32, 1)


def test_empty_range_returns_empty(source01: Path):
    info = probe(source01)
    fb = extract_frames(info, fps=1.0, size=32, start=5.0, end=5.0)
    assert len(fb) == 0
