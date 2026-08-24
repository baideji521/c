"""评价指标测试。"""

from __future__ import annotations

import pytest

from source_trace.evaluation.ground_truth import GroundTruth, GTSegment, PredSegment
from source_trace.evaluation.metrics import evaluate


def _gt(segs) -> GroundTruth:
    return GroundTruth(query="F.mp4", query_duration=20.0, sources=["a.mp4", "b.mp4"], segments=segs)


def test_perfect_match():
    gt = _gt([GTSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0)])
    pred = [PredSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0, 0.99, "HIGH")]
    ev = evaluate(gt, pred)
    assert ev.source_recall == 1.0
    assert ev.segment_recall == 1.0
    assert ev.mean_start_error == pytest.approx(0.0)
    assert ev.median_iou == pytest.approx(1.0)
    assert ev.false_positive_rate == 0.0


def test_wrong_source_counts_as_false_positive():
    gt = _gt([GTSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0)])
    pred = [PredSegment(1, 0.0, 4.0, "b.mp4", 10.0, 14.0, 0.9, "HIGH")]
    ev = evaluate(gt, pred)
    assert ev.source_recall == 0.0
    assert ev.false_positive_rate == 1.0


def test_time_offset_reduces_iou():
    gt = _gt([GTSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0)])
    pred = [PredSegment(1, 0.0, 4.0, "a.mp4", 12.0, 16.0, 0.9, "HIGH")]
    ev = evaluate(gt, pred)
    assert ev.source_recall == 1.0
    assert ev.segment_recall == 0.0  # IoU = 2/6 < 0.5
    assert ev.mean_start_error == pytest.approx(2.0)
    assert ev.median_iou == pytest.approx(1 / 3, abs=1e-6)


def test_missing_prediction():
    gt = _gt([GTSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0)])
    ev = evaluate(gt, [])
    assert ev.source_recall == 0.0
    assert ev.details[0].status == "MISS"


def test_unknown_expected_is_rewarded():
    gt = _gt([GTSegment(1, 0.0, 4.0, "_Hidden.mp4", 10.0, 14.0)])
    ok = evaluate(gt, [PredSegment(1, 0.0, 4.0, None, None, None, 0.2, "UNKNOWN")])
    bad = evaluate(gt, [PredSegment(1, 0.0, 4.0, "a.mp4", 1.0, 5.0, 0.7, "HIGH")])
    assert ok.unknown_accuracy == 1.0
    assert bad.unknown_accuracy == 0.0


def test_aligned_start_error_ignores_shot_boundary_shift():
    """预测的成片边界略有偏移，但线性映射正确时，对齐误差应接近 0。"""
    gt = _gt([GTSegment(1, 2.0, 6.0, "a.mp4", 20.0, 24.0)])
    pred = [PredSegment(1, 1.5, 6.0, "a.mp4", 19.5, 24.0, 0.95, "HIGH", speed=1.0)]
    ev = evaluate(gt, pred)
    assert ev.mean_start_error == pytest.approx(0.5)
    assert ev.mean_aligned_start_error == pytest.approx(0.0, abs=1e-6)


def test_repeated_source_segments():
    gt = _gt(
        [
            GTSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0),
            GTSegment(2, 4.0, 8.0, "b.mp4", 30.0, 34.0),
            GTSegment(3, 8.0, 12.0, "a.mp4", 50.0, 54.0),
        ]
    )
    pred = [
        PredSegment(1, 0.0, 4.0, "a.mp4", 10.0, 14.0, 0.9, "HIGH"),
        PredSegment(2, 4.0, 8.0, "b.mp4", 30.0, 34.0, 0.9, "HIGH"),
        PredSegment(3, 8.0, 12.0, "a.mp4", 50.0, 54.0, 0.9, "HIGH"),
    ]
    ev = evaluate(gt, pred)
    assert ev.source_recall == 1.0 and ev.segment_recall == 1.0
