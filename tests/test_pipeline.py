"""端到端流水线测试（真实跑完整 pipeline，不使用 mock）。

使用 colorhist 描述子以保证无网络依赖；ISC 权重就绪时可用
``SOURCE_TRACE_TEST_MODEL=isc21`` 切换。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from source_trace.config import SourceTraceConfig
from source_trace.evaluation.ground_truth import GroundTruth, PredSegment
from source_trace.evaluation.metrics import evaluate
from source_trace.export.ffmpeg import clip_name
from source_trace.pipeline import run_trace

MODEL = os.environ.get("SOURCE_TRACE_TEST_MODEL", "colorhist")


@pytest.fixture(scope="module")
def traced(basic_dataset: Path, tmp_path_factory):
    out = tmp_path_factory.mktemp("trace_basic")
    cfg = SourceTraceConfig()
    cfg.feature.model = MODEL
    cfg.export.compare_video = False  # 加速测试
    res = run_trace(
        query=basic_dataset / "Final.mp4",
        sources=basic_dataset / "Sources",
        output=out,
        cfg=cfg,
        export_clips=True,
        verify=True,
    )
    return res, basic_dataset


def test_all_segments_resolved(traced):
    res, data = traced
    gt = GroundTruth.load(data / "ground_truth.json")
    assert len(res.segments) == len(gt.segments)
    assert all(s.status != "UNKNOWN" for s in res.segments)


def test_metrics_against_ground_truth(traced):
    res, data = traced
    gt = GroundTruth.load(data / "ground_truth.json")
    preds = [PredSegment.from_dict(s.to_dict()) for s in res.segments]
    ev = evaluate(gt, preds)
    assert ev.source_recall == 1.0
    assert ev.segment_recall == 1.0
    assert ev.mean_start_error < 0.5
    assert ev.mean_end_error < 0.5
    assert ev.median_iou > 0.85
    assert ev.false_positive_rate == 0.0


def test_json_report_schema(traced):
    res, _ = traced
    data = json.loads(res.json_path.read_text(encoding="utf-8"))
    assert data["query"] == "Final.mp4"
    assert len(data["sources"]) == 3
    for seg in data["segments"]:
        for key in ("id", "query_start", "query_end", "source", "source_start", "source_end", "confidence", "status"):
            assert key in seg
        # 需求书要求 visual_score / temporal_score 直接出现在片段顶层
        assert "visual_score" in seg and "temporal_score" in seg
        assert seg["status"] in ("HIGH", "MEDIUM", "LOW", "UNKNOWN")
    assert "timings" in data
    assert data["meta"]["descriptor"]["modality"] == "image"


def test_output_layout(traced):
    """输出目录结构需符合需求书：result.json / result.csv / segments/ / verification/。"""
    res, _ = traced
    out = res.output_dir
    assert (out / "result.json").exists()
    assert (out / "result.csv").exists()
    assert (out / "segments").is_dir()
    assert list((out / "segments").glob("*.mp4"))
    assert (out / "verification").is_dir()


def test_csv_report(traced):
    res, _ = traced
    text = res.csv_path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines[0].startswith("segment_id,query_start,query_end,source,source_start,source_end,confidence,status")
    assert len(lines) == len(res.segments) + 1


def test_clips_exported_with_correct_duration(traced):
    res, _ = traced
    assert res.clips, "没有导出任何片段"
    for c, seg in zip([c for c in res.clips if c.ok], [s for s in res.segments if s.status != "UNKNOWN"]):
        expect = seg.source_end - seg.source_start
        assert abs(c.actual_duration - expect) < 0.3, f"{c.path} 时长偏差过大"
        assert Path(c.path).exists()


def test_clip_naming_convention():
    assert clip_name(1, "Source003.mp4", 52.31, 56.42) == "001_Source003_00-52-31_00-56-42.mp4"


def test_clip_naming_carries_rounded_centiseconds():
    """64.995s 必须进位成 01-05-00，而不是 01-04-100。"""
    assert clip_name(5, "Source01.mp4", 60.515, 64.995) == "005_Source01_01-00-52_01-05-00.mp4"


def test_verification_artifacts(traced):
    res, _ = traced
    sheets = list((res.output_dir / "verification").glob("segment_*.jpg"))
    assert len(sheets) == len(res.segments)
    assert all(p.stat().st_size > 1000 for p in sheets)


def test_cache_is_reused_on_second_run(basic_dataset: Path, tmp_path):
    """同一目录二次运行必须命中缓存，特征提取耗时显著下降。"""
    cfg = SourceTraceConfig()
    cfg.feature.model = MODEL
    cfg.export.contact_sheet = False
    cfg.export.compare_video = False
    kw = dict(
        query=basic_dataset / "Final.mp4",
        sources=basic_dataset / "Sources",
        cfg=cfg,
        export_clips=False,
        verify=False,
    )
    first = run_trace(output=tmp_path / "run1", **kw)
    second = run_trace(output=tmp_path / "run2", **kw)
    assert second.timings["feature_sources"] < first.timings["feature_sources"] * 0.5


def test_error_recovery_with_corrupt_source(basic_dataset: Path, tmp_path):
    """一个素材损坏时其余素材仍应正常处理，并记录到 errors.json。"""
    src = tmp_path / "Sources"
    src.mkdir()
    for p in (basic_dataset / "Sources").glob("*.mp4"):
        (src / p.name).write_bytes(p.read_bytes())
    (src / "Corrupt.mp4").write_bytes(b"this is not a video" * 100)

    cfg = SourceTraceConfig()
    cfg.feature.model = MODEL
    cfg.export.contact_sheet = False
    cfg.export.compare_video = False
    res = run_trace(
        query=basic_dataset / "Final.mp4",
        sources=src,
        output=tmp_path / "out",
        cfg=cfg,
        export_clips=False,
        verify=False,
    )
    assert any("Corrupt.mp4" in (e.get("source") or "") for e in res.errors)
    assert (tmp_path / "out" / "errors.json").exists()
    assert sum(1 for s in res.segments if s.status != "UNKNOWN") >= 2
