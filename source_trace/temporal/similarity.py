"""相似度矩阵构建与分数归一化。

时序对齐的输入统一为帧-帧余弦相似度矩阵 ``S[i, j]``：
    i -> query 采样帧序号（对应 query_ts[i]）
    j -> source 采样帧序号（对应 source_ts[j]）

分数归一化（Score Normalization，借鉴 Meta AI Video Similarity Challenge 方案）：
用「与背景样本的相似度」对原始相似度做校准，压制那些对所有素材都天生相似的帧，
从而降低干扰素材造成的误匹配。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimMatrix:
    matrix: np.ndarray  # (Q, S) float32
    query_ts: np.ndarray  # (Q,) float64
    source_ts: np.ndarray  # (S,) float64
    normalized: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.matrix.shape  # type: ignore[return-value]


def build_sim_matrix(
    query_desc: np.ndarray,
    query_ts: np.ndarray,
    source_desc: np.ndarray,
    source_ts: np.ndarray,
) -> SimMatrix:
    m = query_desc.astype(np.float32) @ source_desc.astype(np.float32).T
    return SimMatrix(matrix=m, query_ts=np.asarray(query_ts, np.float64), source_ts=np.asarray(source_ts, np.float64))


def score_normalize_with_desc(
    sim: SimMatrix, query_desc: np.ndarray, background: np.ndarray, k: int = 10, beta: float = 1.0
) -> SimMatrix:
    """基于背景样本的相似度归一化（需要 query 描述子）。"""
    if background is None or background.size == 0 or sim.matrix.size == 0:
        return sim
    kk = int(min(k, background.shape[0]))
    bg_sim = query_desc.astype(np.float32) @ background.astype(np.float32).T  # (Q, B)
    part = -np.partition(-bg_sim, kk - 1, axis=1)[:, :kk]
    bias = part.mean(axis=1, keepdims=True)  # (Q, 1)
    m = sim.matrix - beta * bias
    return SimMatrix(matrix=m.astype(np.float32), query_ts=sim.query_ts, source_ts=sim.source_ts, normalized=True)


def sample_background(index, exclude_source_id: int, n: int = 256, seed: int = 0) -> np.ndarray:
    """从其它素材中随机采样背景描述子。"""
    rng = np.random.default_rng(seed)
    pools = [e.features.descriptors for e in index.entries if e.source_id != exclude_source_id]
    if not pools:
        return np.zeros((0, 0), dtype=np.float32)
    allv = np.concatenate(pools, axis=0)
    if allv.shape[0] <= n:
        return allv
    idx = rng.choice(allv.shape[0], size=n, replace=False)
    return allv[idx]
