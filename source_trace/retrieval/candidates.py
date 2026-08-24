"""镜头级候选生成。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import RetrievalConfig
from ..features.cache import FeatureSet
from ..utils.log import get_logger
from ..video.shot_detector import Shot
from .index import SourceIndex
from .search import rank_candidates


@dataclass
class ShotCandidates:
    shot: Shot
    query_desc: np.ndarray
    query_ts: np.ndarray
    candidates: list[dict] = field(default_factory=list)

    @property
    def margin(self) -> float:
        """最佳候选与次佳候选的视觉分差（歧义判定用）。"""
        if len(self.candidates) < 2:
            return 1.0 if self.candidates else 0.0
        return float(self.candidates[0]["visual"] - self.candidates[1]["visual"])


def candidates_for_shots(
    index: SourceIndex,
    query_features: FeatureSet,
    shots: list[Shot],
    cfg: RetrievalConfig,
) -> list[ShotCandidates]:
    """为每个镜头生成候选素材列表。"""
    log = get_logger()
    top_k = cfg.top_k if index.size >= cfg.top_k else index.size
    out: list[ShotCandidates] = []
    for shot in shots:
        sub = query_features.slice_time(shot.start, shot.end)
        if len(sub) == 0:
            # 镜头过短导致 1FPS 采样为空：取最近的一帧
            i = int(np.argmin(np.abs(query_features.timestamps - shot.mid)))
            desc = query_features.descriptors[i : i + 1]
            ts = query_features.timestamps[i : i + 1]
        else:
            desc, ts = sub.descriptors, sub.timestamps

        cands = rank_candidates(
            index, desc, top_k=top_k, frame_top_n=cfg.frame_top_n, min_source_sim=cfg.min_source_sim
        )
        out.append(ShotCandidates(shot=shot, query_desc=desc, query_ts=ts, candidates=cands))
        log.info(
            "Segment %02d [%.2f-%.2f] 候选：%s",
            shot.shot_id, shot.start, shot.end,
            "  ".join(f"{c['name']} {c['visual']:.3f}" for c in cands),
        )
    return out
