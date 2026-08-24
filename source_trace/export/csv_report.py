"""CSV 报告输出。"""

from __future__ import annotations

import csv
from pathlib import Path

from ..temporal.mapping import TimeSegment

FIELDS = [
    "segment_id",
    "query_start",
    "query_end",
    "source",
    "source_start",
    "source_end",
    "confidence",
    "status",
    "speed",
    "visual_score",
    "temporal_score",
    "margin_score",
    "alignment_score",
]


def write_csv(segments: list[TimeSegment], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for s in segments:
            w.writerow(
                {
                    "segment_id": s.id,
                    "query_start": f"{s.query_start:.3f}",
                    "query_end": f"{s.query_end:.3f}",
                    "source": s.source or "",
                    "source_start": "" if s.source_start is None else f"{s.source_start:.3f}",
                    "source_end": "" if s.source_end is None else f"{s.source_end:.3f}",
                    "confidence": f"{s.confidence:.4f}",
                    "status": s.status,
                    "speed": f"{s.speed:.4f}",
                    "visual_score": f"{s.scores.get('visual_score', 0.0):.4f}",
                    "temporal_score": f"{s.scores.get('temporal_score', 0.0):.4f}",
                    "margin_score": f"{s.scores.get('margin_score', 0.0):.4f}",
                    "alignment_score": f"{s.scores.get('alignment_score', 0.0):.4f}",
                }
            )
    return path
