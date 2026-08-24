"""Ground Truth 读取与预测结果对齐。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GTSegment:
    id: int
    query_start: float
    query_end: float
    source: str
    source_start: float
    source_end: float
    speed: float = 1.0
    transforms: dict | None = None
    note: str = ""

    @property
    def is_unknown_expected(self) -> bool:
        """来源素材未提供给算法时，期望输出 UNKNOWN。"""
        return self.source.startswith("_Hidden")


@dataclass
class GroundTruth:
    query: str
    query_duration: float
    sources: list[str]
    segments: list[GTSegment]
    case: str = ""

    @classmethod
    def load(cls, path: Path | str) -> "GroundTruth":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        segs = [
            GTSegment(
                id=int(s.get("id", i + 1)),
                query_start=float(s["query_start"]),
                query_end=float(s["query_end"]),
                source=str(s["source"]),
                source_start=float(s["source_start"]),
                source_end=float(s["source_end"]),
                speed=float(s.get("speed", 1.0)),
                transforms=s.get("transforms"),
                note=str(s.get("note", "")),
            )
            for i, s in enumerate(data.get("segments", []))
        ]
        return cls(
            query=str(data.get("query", "")),
            query_duration=float(data.get("query_duration", 0.0)),
            sources=list(data.get("sources", [])),
            segments=segs,
            case=str(data.get("case", "")),
        )


@dataclass
class PredSegment:
    id: int
    query_start: float
    query_end: float
    source: str | None
    source_start: float | None
    source_end: float | None
    confidence: float
    status: str
    speed: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "PredSegment":
        return cls(
            id=int(d["id"]),
            query_start=float(d["query_start"]),
            query_end=float(d["query_end"]),
            source=d.get("source"),
            source_start=None if d.get("source_start") is None else float(d["source_start"]),
            source_end=None if d.get("source_end") is None else float(d["source_end"]),
            confidence=float(d.get("confidence", 0.0)),
            status=str(d.get("status", "UNKNOWN")),
            speed=float(d.get("speed", 1.0)),
        )


def load_predictions(path: Path | str) -> list[PredSegment]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [PredSegment.from_dict(s) for s in data.get("segments", [])]
