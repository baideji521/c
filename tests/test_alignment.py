"""时序对齐与时间映射测试（使用构造的相似度矩阵，结果完全确定）。"""

from __future__ import annotations

import numpy as np
import pytest

from source_trace.config import AlignmentConfig, ConfidenceConfig
from source_trace.temporal.alignment import align
from source_trace.temporal.mapping import (
    TimeSegment,
    compute_confidence,
    decide_status,
    merge_segments,
    path_to_time,
    robust_time_fit,
)


def _make_sim(q: int, s: int, offset: int, slope: float = 1.0, noise: float = 0.1, seed: int = 0) -> np.ndarray:
    """构造一条 source = slope*query + offset 的高相似度对角线。"""
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0.0, noise, size=(q, s)).astype(np.float32)
    for i in range(q):
        j = int(round(slope * i + offset))
        if 0 <= j < s:
            sim[i, j] = 0.95
            if j + 1 < s:
                sim[i, j + 1] = max(sim[i, j + 1], 0.6)
    return sim


@pytest.mark.parametrize("method", ["tn", "dp", "dtw", "hv"])
def test_align_finds_diagonal(method: str):
    cfg = AlignmentConfig(method=method)
    sim = _make_sim(q=10, s=60, offset=30)
    paths = align(sim, cfg)
    assert paths, f"{method} 未找到任何路径"
    p = paths[0]
    assert p.q_start <= 1 and p.q_end >= 8
    assert abs(p.s_start - 30) <= 2
    assert p.slope == pytest.approx(1.0, abs=0.15)


def test_tn_rejects_random_matrix():
    """纯噪声矩阵不应产生高覆盖率的长路径。"""
    cfg = AlignmentConfig(method="tn")
    rng = np.random.default_rng(7)
    sim = rng.uniform(0.0, 0.3, size=(10, 60)).astype(np.float32)
    paths = align(sim, cfg)
    if paths:
        # 允许找到短路径，但视觉分必须明显偏低
        assert paths[0].visual < 0.45


def test_tn_multi_segment():
    """一个 query 段落分别对应 source 两个不同区间时，TN 应给出多条路径。"""
    cfg = AlignmentConfig(method="tn", min_length=3)
    sim = np.random.default_rng(1).uniform(0, 0.1, size=(12, 80)).astype(np.float32)
    for i in range(6):
        sim[i, 10 + i] = 0.95
    for i in range(6, 12):
        sim[i, 50 + (i - 6)] = 0.95
    paths = align(sim, cfg, max_paths=4)
    assert len(paths) >= 2
    starts = sorted(int(p.s_start) for p in paths[:2])
    assert abs(starts[0] - 10) <= 2 and abs(starts[1] - 50) <= 2


def test_speed_change_slope_detected():
    """1.25 倍速：source 时间轴步长应为 query 的 1.25 倍。"""
    cfg = AlignmentConfig(method="tn", min_slope=0.5, max_slope=2.0)
    sim = _make_sim(q=12, s=80, offset=20, slope=1.25)
    paths = align(sim, cfg)
    assert paths
    assert paths[0].slope == pytest.approx(1.25, abs=0.2)


def test_robust_time_fit_ignores_outliers():
    cfg = AlignmentConfig()
    q = np.arange(10, dtype=np.float64)
    s = 2.0 + q  # 完美对应
    s[3] = 40.0  # 离群点
    fit = robust_time_fit(q, s, cfg)
    assert fit.slope == pytest.approx(1.0, abs=0.05)
    assert fit.intercept == pytest.approx(2.0, abs=0.15)
    assert fit.rmse < 0.2
    assert fit.n_points >= 8


def test_path_to_time_extrapolates_boundaries():
    """匹配点按 1FPS 采样，但输出时间应精确外推到镜头边界。"""
    cfg = AlignmentConfig(method="tn")
    sim = _make_sim(q=5, s=60, offset=30)
    paths = align(sim, cfg)
    assert paths
    query_ts = np.arange(5, dtype=np.float64) + 10.0  # 10,11,...,14
    source_ts = np.arange(60, dtype=np.float64)
    s0, s1, fit = path_to_time(
        paths[0], query_ts, source_ts, query_start=9.6, query_end=14.4, cfg=cfg, source_duration=60.0
    )
    # query 10 -> source 30，故 query 9.6 -> 29.6
    assert s0 == pytest.approx(29.6, abs=0.15)
    assert s1 == pytest.approx(34.4, abs=0.15)
    assert fit.slope == pytest.approx(1.0, abs=0.05)


def test_confidence_weights_configurable():
    cfg = ConfidenceConfig()
    fit = robust_time_fit(np.arange(6.0), np.arange(6.0) + 3.0, AlignmentConfig())
    high, _ = compute_confidence(0.95, 1.0, 0.3, fit, 4.0, cfg)
    low, _ = compute_confidence(0.45, 0.3, 0.005, fit, 4.0, cfg)
    assert high > low
    cfg2 = ConfidenceConfig(w_visual=1.0, w_temporal=0.0, w_margin=0.0, w_alignment=0.0)
    only_visual, _ = compute_confidence(0.5, 1.0, 1.0, fit, 10.0, cfg2)
    assert only_visual == pytest.approx(0.5, abs=0.02)


def test_status_unknown_on_ambiguity():
    cfg = ConfidenceConfig()
    # 两个候选几乎同分 -> 必须 UNKNOWN，不能硬给来源
    assert decide_status(0.70, margin=0.01, cfg=cfg) == "UNKNOWN"
    assert decide_status(0.90, margin=0.30, cfg=cfg) == "HIGH"
    assert decide_status(0.70, margin=0.30, cfg=cfg) == "MEDIUM"
    assert decide_status(0.55, margin=0.30, cfg=cfg) == "LOW"
    assert decide_status(0.20, margin=0.30, cfg=cfg) == "UNKNOWN"


def _seg(i, qs, qe, src, ss, se, conf=0.9, status="HIGH", speed=1.0) -> TimeSegment:
    return TimeSegment(i, qs, qe, src, 1, ss, se, conf, status, speed=speed, scores={"visual_score": conf})


def test_merge_adjacent_continuous_segments():
    segs = [_seg(1, 0.0, 2.0, "a.mp4", 10.0, 12.0), _seg(2, 2.0, 4.0, "a.mp4", 12.0, 14.0)]
    merged = merge_segments(segs)
    assert len(merged) == 1
    assert merged[0].query_end == 4.0 and merged[0].source_end == 14.0


def test_do_not_merge_time_jump():
    segs = [_seg(1, 0.0, 2.0, "a.mp4", 10.0, 12.0), _seg(2, 2.0, 4.0, "a.mp4", 40.0, 42.0)]
    assert len(merge_segments(segs)) == 2


def test_do_not_merge_different_source():
    segs = [_seg(1, 0.0, 2.0, "a.mp4", 10.0, 12.0), _seg(2, 2.0, 4.0, "b.mp4", 12.0, 14.0)]
    assert len(merge_segments(segs)) == 2


def test_do_not_merge_unknown():
    segs = [
        _seg(1, 0.0, 2.0, "a.mp4", 10.0, 12.0),
        TimeSegment(2, 2.0, 4.0, None, None, None, None, 0.1, "UNKNOWN"),
    ]
    assert len(merge_segments(segs)) == 2
