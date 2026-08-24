"""帧级检索与素材打分。

打分策略（第一轮粗筛，绝不作为最终答案）：
* ``max_mean``：对每个 query 帧取该 source 内的最大相似度，再对所有 query 帧求均值
* ``vote``    ：统计每个 query 帧的全局 Top-N 命中中，各 source 的加权票数

两者结合可同时反映「相似强度」与「命中广度」。
"""

from __future__ import annotations

import numpy as np

from .index import SourceIndex


def source_scores(index: SourceIndex, query_desc: np.ndarray, trim_ratio: float = 0.2) -> dict[int, dict]:
    """对所有 source 打分。

    trim_ratio: 计算均值时截掉最低的一部分 query 帧（成片片段边缘常含转场/黑帧）。
    返回 {source_id: {"max_mean":…, "max_median":…, "hit_ratio":…}}
    """
    out: dict[int, dict] = {}
    if query_desc.shape[0] == 0:
        return out
    for e in index.entries:
        sim = query_desc.astype(np.float32) @ e.features.descriptors.astype(np.float32).T
        if sim.size == 0:
            out[e.source_id] = {"max_mean": 0.0, "max_median": 0.0, "hit_ratio": 0.0}
            continue
        per_frame = sim.max(axis=1)
        k = max(1, int(round(per_frame.size * (1.0 - trim_ratio))))
        trimmed = np.sort(per_frame)[-k:]
        out[e.source_id] = {
            "max_mean": float(trimmed.mean()),
            "max_median": float(np.median(per_frame)),
            "hit_ratio": float((per_frame > 0.5).mean()),
        }
    return out


def frame_votes(index: SourceIndex, query_desc: np.ndarray, top_n: int) -> dict[int, float]:
    """全局 Top-N 命中的加权投票（权重 = 相似度）。"""
    sims, owners, _ = index.search(query_desc, top_n=top_n)
    votes: dict[int, float] = {}
    for row_s, row_o in zip(sims, owners):
        for s, o in zip(row_s, row_o):
            votes[int(o)] = votes.get(int(o), 0.0) + float(max(s, 0.0))
    total = sum(votes.values()) or 1.0
    return {k: v / total for k, v in votes.items()}


def rank_candidates(
    index: SourceIndex,
    query_desc: np.ndarray,
    top_k: int,
    frame_top_n: int,
    min_source_sim: float,
) -> list[dict]:
    """返回按综合得分降序排列的候选素材列表。"""
    scores = source_scores(index, query_desc)
    votes = frame_votes(index, query_desc, top_n=frame_top_n)
    ranked = []
    for sid, s in scores.items():
        vote = votes.get(sid, 0.0)
        combined = 0.75 * s["max_mean"] + 0.25 * vote
        ranked.append(
            {
                "source_id": sid,
                "name": index.get(sid).name,
                "visual": s["max_mean"],
                "median": s["max_median"],
                "hit_ratio": s["hit_ratio"],
                "vote": vote,
                "score": combined,
            }
        )
    ranked.sort(key=lambda d: -d["score"])
    kept = [r for r in ranked if r["visual"] >= min_source_sim][: max(1, top_k)]
    if not kept:  # 全部低于阈值时仍保留最佳者，交由后续时序对齐/置信度判 UNKNOWN
        kept = ranked[:1]
    return kept
