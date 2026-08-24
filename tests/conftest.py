"""pytest 公共 fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASIC = ROOT / "testdata" / "basic"


@pytest.fixture(scope="session")
def basic_dataset() -> Path:
    """basic 合成数据集（不存在时自动生成）。"""
    if not (BASIC / "ground_truth.json").exists():
        from source_trace.evaluation.synthetic import make_dataset

        make_dataset(BASIC, "basic")
    return BASIC


@pytest.fixture(scope="session")
def source01(basic_dataset: Path) -> Path:
    return basic_dataset / "Sources" / "Source01.mp4"
