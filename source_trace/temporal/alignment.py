"""时序对齐（Temporal Alignment）。

这是本项目的核心：把「单帧相似」升级为「时间轴上的连续对应关系」。

四种方法（均为纯 numpy，无需训练、无需 GPU）：

* ``tn``  Temporal Network（VCSL 主力方法的复现）
        在候选匹配点集合上构造有向无环图，边要求 query/source 同时递增且步长受限，
        用最长加权路径求解一条对应轨迹；移除已用点后迭代，天然支持多片段。
* ``dp``  动态规划（类 Smith-Waterman 局部对齐），带 gap 惩罚。
* ``dtw`` 动态时间规整，允许非线性速度变化。
* ``hv``  Hough Voting，对时间偏移 (source_t - k*query_t) 直方图投票，抗噪、极快。

统一输出 ``AlignPath``：帧序号区间 + 匹配点 + 斜率 k（变速倍率）+ 各项子分数。
时间单位的换算由 ``mapping.py`` 完成（严格使用 timestamp，不用 frame index）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AlignmentConfig


@dataclass
class AlignPath:
    """一条时序对应路径（帧序号域）。"""

    q_start: int
    q_end: int
    s_start: int
    s_end: int
    points: np.ndarray = field(repr=False)  # (M, 2) int，[query_idx, source_idx]
    sims: np.ndarray = field(repr=False)  # (M,) float
    slope: float = 1.0
    intercept: float = 0.0
    visual: float = 0.0  # 路径点平均相似度
    coverage: float = 0.0  # query 区间被路径点覆盖的比例
    linearity: float = 0.0  # 线性拟合优度（0~1）
    method: str = "tn"

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def q_len(self) -> int:
        return self.q_end - self.q_start + 1

    def to_dict(self) -> dict:
        return {
            "q_start": self.q_start,
            "q_end": self.q_end,
            "s_start": self.s_start,
            "s_end": self.s_end,
            "n_points": self.n_points,
            "slope": round(self.slope, 4),
            "visual": round(self.visual, 4),
            "coverage": round(self.coverage, 4),
            "linearity": round(self.linearity, 4),
            "method": self.method,
        }


# ---------------------------------------------------------------- 公共工具


def _effective_min_sim(sim: np.ndarray, cfg: AlignmentConfig) -> float:
    """自适应下限：固定阈值与「全矩阵前 keep_mult*Q 高分」两者取大。"""
    thr = float(cfg.min_sim)
    if not cfg.adaptive_min_sim or sim.size == 0:
        return thr
    q = sim.shape[0]
    k = int(min(sim.size, max(1, np.ceil(cfg.adaptive_keep_mult * q))))
    kth = float(np.partition(sim.ravel(), -k)[-k])
    return max(thr, kth)


def _candidate_points(sim: np.ndarray, min_sim: float, top_k: int, floor: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """候选匹配点选取。

    * 每个 query 帧取 top_k 个最高分，且分数需 >= ``min_sim``（自适应下限）
    * 同时保证每个 query 帧的 top-1 只要超过基础阈值 ``floor`` 就被保留，
      避免自适应下限过严时某些帧完全没有候选点，导致时序路径断裂
    """
    q, s = sim.shape
    top_k = int(min(max(1, top_k), s))
    idx = np.argpartition(-sim, top_k - 1, axis=1)[:, :top_k]
    vals = np.take_along_axis(sim, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    vals = np.take_along_axis(vals, order, axis=1)

    mask = vals >= min_sim
    if floor is not None:
        keep_top1 = vals[:, 0] >= floor
        mask[:, 0] |= keep_top1

    qi = np.repeat(np.arange(q), top_k)[mask.ravel()]
    si = idx.ravel()[mask.ravel()]
    sv = vals.ravel()[mask.ravel()]
    order2 = np.lexsort((si, qi))
    pts = np.stack([qi[order2], si[order2]], axis=1).astype(np.int32)
    return pts, sv[order2].astype(np.float32)


def _fit_line(points: np.ndarray) -> tuple[float, float, float]:
    """最小二乘拟合 source = k * query + b，返回 (k, b, R²)。"""
    if points.shape[0] < 2:
        return 1.0, float(points[0, 1] - points[0, 0]) if points.size else 0.0, 1.0
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    if np.ptp(x) < 1e-9:
        return 1.0, float(y.mean() - x.mean()), 0.0
    k, b = np.polyfit(x, y, 1)
    pred = k * x + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return float(k), float(b), float(max(0.0, min(1.0, r2)))


def _finalize(points: np.ndarray, sims: np.ndarray, sim: np.ndarray, method: str) -> AlignPath:
    k, b, r2 = _fit_line(points)
    q0, q1 = int(points[:, 0].min()), int(points[:, 0].max())
    s0, s1 = int(points[:, 1].min()), int(points[:, 1].max())
    q_span = q1 - q0 + 1
    coverage = float(np.unique(points[:, 0]).size) / max(1, q_span)
    return AlignPath(
        q_start=q0,
        q_end=q1,
        s_start=s0,
        s_end=s1,
        points=points,
        sims=sims,
        slope=k,
        intercept=b,
        visual=float(sims.mean()),
        coverage=min(1.0, coverage),
        linearity=r2,
        method=method,
    )


# ---------------------------------------------------------------- TN


def temporal_network(sim: np.ndarray, cfg: AlignmentConfig, max_paths: int = 6) -> list[AlignPath]:
    """Temporal Network：候选点建图 + 最长加权路径，迭代提取多条路径。

    与 VCSL 参考实现（alipay/VCSL ``vcsl/vta.py`` 的 ``tn()``）的对应关系：
      - 相同：每行取 top-k 候选点建 DAG；边要求 query / refer 同时严格递增且步长
        不超过 ``tn_max_step``；取最长加权路径；迭代提取多段以支持一条成片用到
        同一素材的多个区间。
      - 不同 1：VCSL 用 networkx ``dag_longest_path`` + 虚拟 source/sink，这里用
        按 (q, s) 字典序排序后的 numpy DP，避免引入 networkx 依赖，复杂度相同。
      - 不同 2：额外加入斜率约束 ``[min_slope, max_slope]``（VCSL 没有）。本工具要处理
        0.8x~1.25x 变速，斜率即变速倍率，把它约束在合理区间能直接排掉大量伪路径。
      - 不同 3：VCSL 用候选框 IoU（``max_iou``）过滤重复路径；这里改为「移除已用
        query 行」。成片的一个时间点只可能来自一个素材区间，按 query 行去重更贴合
        本任务，且能防止同一段被反复输出。
    """
    pts, vals = _candidate_points(sim, _effective_min_sim(sim, cfg), cfg.tn_top_k, floor=cfg.min_sim)
    if pts.shape[0] == 0:
        return []

    paths: list[AlignPath] = []
    alive = np.ones(pts.shape[0], dtype=bool)

    for _ in range(max_paths):
        idx_alive = np.flatnonzero(alive)
        if idx_alive.size < cfg.min_length:
            break
        P = pts[idx_alive]
        V = vals[idx_alive]
        n = P.shape[0]

        # DP：best[i] = 以 i 结尾的最大累计得分（点按 (q, s) 升序）
        best = V.astype(np.float64).copy()
        prev = np.full(n, -1, dtype=np.int32)
        step = int(cfg.tn_max_step)

        for i in range(n):
            qi, si = P[i]
            # 只需回看 query 距离在 step 内的点；点已排序，用二分定位
            lo = np.searchsorted(P[:, 0], qi - step, side="left")
            for j in range(lo, i):
                qj, sj = P[j]
                if qj >= qi and sj >= si:
                    continue
                dq = qi - qj
                ds = si - sj
                if dq <= 0 or ds <= 0:
                    continue
                if dq > step or ds > step:
                    continue
                slope = ds / dq
                if slope < cfg.min_slope or slope > cfg.max_slope:
                    continue
                cand = best[j] + V[i]
                if cand > best[i]:
                    best[i] = cand
                    prev[i] = j

        end = int(np.argmax(best))
        chain: list[int] = []
        cur = end
        while cur != -1:
            chain.append(cur)
            cur = int(prev[cur])
        chain.reverse()
        if len(chain) < cfg.min_length:
            break

        sel = idx_alive[chain]
        path = _finalize(pts[sel], vals[sel], sim, "tn")
        paths.append(path)

        # 移除已使用的 query 行，避免重复输出同一段
        used_q = set(int(x) for x in pts[sel][:, 0])
        alive[np.isin(pts[:, 0], list(used_q))] = False

    paths.sort(key=lambda p: -(p.visual * p.n_points))
    return paths


# ---------------------------------------------------------------- DP


def dynamic_programming(sim: np.ndarray, cfg: AlignmentConfig, max_paths: int = 4) -> list[AlignPath]:
    """局部对齐 DP（Smith-Waterman 变体）：match=sim-min_sim，gap 受惩罚。"""
    q, s = sim.shape
    if q == 0 or s == 0:
        return []
    score = (sim - _effective_min_sim(sim, cfg)).astype(np.float64)
    gp = float(cfg.dp_gap_penalty)

    H = np.zeros((q + 1, s + 1), dtype=np.float64)
    ptr = np.zeros((q + 1, s + 1), dtype=np.int8)  # 0=stop 1=diag 2=up 3=left
    for i in range(1, q + 1):
        for j in range(1, s + 1):
            diag = H[i - 1, j - 1] + score[i - 1, j - 1]
            up = H[i - 1, j] - gp
            left = H[i, j - 1] - gp
            m = max(0.0, diag, up, left)
            H[i, j] = m
            ptr[i, j] = 0 if m == 0.0 else (1 if m == diag else (2 if m == up else 3))

    paths: list[AlignPath] = []
    Hw = H.copy()
    for _ in range(max_paths):
        i, j = np.unravel_index(int(np.argmax(Hw)), Hw.shape)
        if Hw[i, j] <= 0:
            break
        pts: list[tuple[int, int]] = []
        sims: list[float] = []
        ci, cj = int(i), int(j)
        while ci > 0 and cj > 0 and ptr[ci, cj] != 0:
            if ptr[ci, cj] == 1:
                pts.append((ci - 1, cj - 1))
                sims.append(float(sim[ci - 1, cj - 1]))
                ci, cj = ci - 1, cj - 1
            elif ptr[ci, cj] == 2:
                ci -= 1
            else:
                cj -= 1
        if len(pts) < cfg.min_length:
            Hw[i, j] = 0.0
            continue
        arr = np.array(pts[::-1], dtype=np.int32)
        paths.append(_finalize(arr, np.array(sims[::-1], dtype=np.float32), sim, "dp"))
        Hw[arr[:, 0].min() + 1 : arr[:, 0].max() + 2, :] = 0.0
    return paths


# ---------------------------------------------------------------- DTW


def dtw_align(sim: np.ndarray, cfg: AlignmentConfig, max_paths: int = 1) -> list[AlignPath]:
    """DTW：全局规整后按相似度阈值截取有效子段。"""
    q, s = sim.shape
    if q == 0 or s == 0:
        return []
    cost = 1.0 - sim.astype(np.float64)
    D = np.full((q + 1, s + 1), np.inf)
    D[0, :] = 0.0  # 允许 source 任意起点（子序列 DTW）
    for i in range(1, q + 1):
        for j in range(1, s + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])

    j = int(np.argmin(D[q, 1:])) + 1
    i = q
    pts: list[tuple[int, int]] = []
    sims: list[float] = []
    while i > 0 and j > 0:
        pts.append((i - 1, j - 1))
        sims.append(float(sim[i - 1, j - 1]))
        choices = (D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
        step = int(np.argmin(choices))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    arr = np.array(pts[::-1], dtype=np.int32)
    svals = np.array(sims[::-1], dtype=np.float32)
    keep = svals >= _effective_min_sim(sim, cfg)
    if keep.sum() < cfg.min_length:
        return []
    return [_finalize(arr[keep], svals[keep], sim, "dtw")]


# ---------------------------------------------------------------- Hough Voting


def hough_voting(sim: np.ndarray, cfg: AlignmentConfig, max_paths: int = 4) -> list[AlignPath]:
    """Hough 投票：在多个速度假设下对时间偏移做直方图投票，取峰值。"""
    pts, vals = _candidate_points(sim, _effective_min_sim(sim, cfg), cfg.tn_top_k, floor=cfg.min_sim)
    if pts.shape[0] == 0:
        return []
    q_n, s_n = sim.shape
    slopes = np.array([0.8, 0.9, 1.0, 1.1, 1.25], dtype=np.float64)
    slopes = slopes[(slopes >= cfg.min_slope) & (slopes <= cfg.max_slope)]
    if slopes.size == 0:
        slopes = np.array([1.0])

    paths: list[AlignPath] = []
    used_q: set[int] = set()
    for _ in range(max_paths):
        best = None
        for k in slopes:
            offset = pts[:, 1] - k * pts[:, 0]
            lo, hi = float(offset.min()), float(offset.max())
            nbins = max(4, int(np.ceil(hi - lo)) + 1)
            hist, edges = np.histogram(offset, bins=nbins, range=(lo, hi + 1e-6), weights=vals)
            b = int(np.argmax(hist))
            sel = (offset >= edges[b] - 1.0) & (offset <= edges[b + 1] + 1.0)
            sel &= ~np.isin(pts[:, 0], list(used_q)) if used_q else sel
            if sel.sum() < cfg.min_length:
                continue
            weight = float(vals[sel].sum())
            if best is None or weight > best[0]:
                best = (weight, sel, k)
        if best is None:
            break
        _, sel, _k = best
        path = _finalize(pts[sel], vals[sel], sim, "hv")
        paths.append(path)
        used_q |= set(int(x) for x in pts[sel][:, 0])
        if len(used_q) >= q_n:
            break
    paths.sort(key=lambda p: -(p.visual * p.n_points))
    return paths


# ---------------------------------------------------------------- 入口

_METHODS = {
    "tn": temporal_network,
    "dp": dynamic_programming,
    "dtw": dtw_align,
    "hv": hough_voting,
}


def align(sim: np.ndarray, cfg: AlignmentConfig, method: str | None = None, max_paths: int = 6) -> list[AlignPath]:
    """按配置执行时序对齐。method=none 时退化为「每帧独立最近邻」（仅用于 benchmark 对照）。"""
    m = (method or cfg.method).lower()
    if m == "none":
        return _nearest_only(sim, cfg)
    fn = _METHODS.get(m)
    if fn is None:
        raise ValueError(f"未知的时序对齐方法：{m}（可选 {sorted(_METHODS) + ['none']}）")
    return fn(sim, cfg, max_paths=max_paths)


def _nearest_only(sim: np.ndarray, cfg: AlignmentConfig) -> list[AlignPath]:
    """无时序约束的基线：每个 query 帧取全局最相似帧。"""
    if sim.size == 0:
        return []
    j = sim.argmax(axis=1)
    v = sim.max(axis=1)
    keep = v >= min(cfg.min_sim, _effective_min_sim(sim, cfg))
    if keep.sum() == 0:
        return []
    pts = np.stack([np.flatnonzero(keep), j[keep]], axis=1).astype(np.int32)
    return [_finalize(pts, v[keep].astype(np.float32), sim, "none")]
