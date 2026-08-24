"""镜头检测测试。"""

from __future__ import annotations

from pathlib import Path

from source_trace.config import ShotConfig
from source_trace.video.probe import probe
from source_trace.video.shot_detector import detect_shots


def test_detects_three_shots_in_basic_final(basic_dataset: Path):
    """basic 用例的成片由 3 段拼接（4.00 / 8.48 处切换）。"""
    info = probe(basic_dataset / "Final.mp4")
    shots = detect_shots(info, ShotConfig(method="histogram"))
    assert 3 <= len(shots) <= 5
    bounds = [s.start for s in shots[1:]]
    for expect in (4.0, 8.48):
        assert any(abs(b - expect) < 0.4 for b in bounds), f"未在 {expect}s 附近检测到镜头切换：{bounds}"


def test_shots_cover_full_duration_without_gap(basic_dataset: Path):
    info = probe(basic_dataset / "Final.mp4")
    shots = detect_shots(info, ShotConfig(method="histogram"))
    assert shots[0].start == 0.0
    assert abs(shots[-1].end - info.duration) < 0.05
    for a, b in zip(shots, shots[1:]):
        assert abs(b.start - a.end) < 1e-6


def test_fixed_window_method(basic_dataset: Path):
    info = probe(basic_dataset / "Final.mp4")
    shots = detect_shots(info, ShotConfig(method="fixed", fixed_window_sec=2.0, min_shot_sec=0.5))
    assert len(shots) >= 6
    assert all(s.duration <= 2.5 for s in shots)


def test_max_shot_split(basic_dataset: Path):
    info = probe(basic_dataset / "Final.mp4")
    shots = detect_shots(info, ShotConfig(method="histogram", max_shot_sec=2.0))
    assert all(s.duration <= 2.05 for s in shots)


def test_shot_ids_sequential(basic_dataset: Path):
    info = probe(basic_dataset / "Final.mp4")
    shots = detect_shots(info, ShotConfig())
    assert [s.shot_id for s in shots] == list(range(1, len(shots) + 1))
