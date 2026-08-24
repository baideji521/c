"""JSON 报告输出。"""

from __future__ import annotations

import json
from pathlib import Path

from ..temporal.mapping import TimeSegment


def build_result(
    query_name: str,
    query_duration: float,
    sources: list[str],
    segments: list[TimeSegment],
    meta: dict,
) -> dict:
    return {
        "query": query_name,
        "query_duration": round(query_duration, 3),
        "sources": sources,
        "segments": [s.to_dict() for s in segments],
        "summary": {
            "n_segments": len(segments),
            "n_high": sum(1 for s in segments if s.status == "HIGH"),
            "n_medium": sum(1 for s in segments if s.status == "MEDIUM"),
            "n_low": sum(1 for s in segments if s.status == "LOW"),
            "n_unknown": sum(1 for s in segments if s.status == "UNKNOWN"),
            "matched_ratio": round(
                sum(s.query_end - s.query_start for s in segments if s.status != "UNKNOWN")
                / max(query_duration, 1e-9),
                4,
            ),
        },
        "meta": meta,
    }


def write_json(result: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_errors(errors: list[dict], path: Path | str) -> Path | None:
    if not errors:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
