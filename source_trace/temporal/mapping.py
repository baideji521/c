"""帧序号 -> 真实时间 的映射、置信度计算与片段合并。

时间一律使用 timestamp（float64 秒），绝不用 frame index 直接换算。

核心思想：把匹配点在「时间域」上做鲁棒线性拟合
    source_t = k * query_t + b
其中 k 即变速倍率（0.8/0.9/1.1/1.25 ...），要求在一个片段内保持稳定。
再用该直线把成片片段的起止时间外推到原始素材时间，
因此定位精度可以优于采样间隔本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import AlignmentConfig, ConfidenceConfig
from .alignment import AlignPath


@dataclass
class TimeSegment:
    """最终输出的一个反溯片段。"""

    id: int
    query_start: float
    query_end: float
    source: str | None
    source_id: int | None
    source_start: float | None
    source_end: float | None
    confidence: float
    status: str  # HIGH / MEDIUM / LOW / UNKNOWN
    speed: float = 1.0
    scores: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    method: str = "tn"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "query_start": round(self.query_start, 3),
            "query_end": round(self.query_end, 3),
            "source": self.source,
            "source_start": None if self.source_start is None else round(self.source_start, 3),
            "source_end": None if self.source_end is None else round(self.source_end, 3),
            "confidence": round(self.confidence, 4),
            "status": self.status,
            "speed": round(self.speed, 4),
            # 顶层再暴露两个主分数便于直接读取，完整明细仍在 scores 中
            "visual_score": round(float(self.scores.get("visual_score", 0.0)), 4),
            "temporal_score": round(float(self.scores.get("temporal_score", 0.0)), 4),
            "scores": {k: round(float(v), 4) for k, v in self.scores.items()},
            "candidates": [
                {"source": c.get("name"), "visual": round(float(c.get("visual", 0.0)), 4)}
                for c in self.candidates[:5]
            ],
            "method": self.method,
        }


@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    rmse: float  # 秒
    inliers: np.ndarray  # bool mask
    n_points: int


def robust_time_fit(
    q_times: np.ndarray, s_times: np.ndarray, cfg: AlignmentConfig, sigma: float = 2.0
) -> FitResult:
    """时间域鲁棒线性拟合（最小二乘 + 残差剔除迭代）。"""
    q = np.asarray(q_times, dtype=np.float64)
    s = np.asarray(s_times, dtype=np.float64)
    n = q.size
    if n == 0:
        return FitResult(1.0, 0.0, 0.0, float("inf"), np.zeros(0, bool), 0)
    if n == 1:
        return FitResult(1.0, float(s[0] - q[0]), 1.0, 0.0, np.ones(1, bool), 1)

    mask = np.ones(n, dtype=bool)
    slope, intercept = 1.0, float(np.median(s - q))
    for _ in range(3):
        qq, ss = q[mask], s[mask]
        if qq.size < 2 or np.ptp(qq) < 1e-9:
            slope, intercept = 1.0, float(np.median(ss - qq)) if qq.size else 0.0
            break
        slope, intercept = np.polyfit(qq, ss, 1)
        slope = float(np.clip(slope, cfg.min_slope, cfg.max_slope))
        intercept = float(np.median(ss - slope * qq))
        resid = np.abs(s - (slope * q + intercept))
        thr = max(float(resid[mask].std() * sigma), 0.15)
        new_mask = resid <= thr
        if new_mask.sum() < max(2, int(0.4 * n)) or (new_mask == mask).all():
            mask = new_mask if new_mask.sum() >= 2 else mask
            break
        mask = new_mask

    qq, ss = q[mask], s[mask]
    pred = slope * qq + intercept
    rmse = float(np.sqrt(((ss - pred) ** 2).mean())) if qq.size else float("inf")
    ss_tot = float(((ss - ss.mean()) ** 2).sum()) if qq.size else 0.0
    ss_res = float(((ss - pred) ** 2).sum()) if qq.size else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else (1.0 if rmse < 0.3 else 0.0)
    return FitResult(
        slope=float(slope),
        intercept=float(intercept),
        r2=float(max(0.0, min(1.0, r2))),
        rmse=rmse,
        inliers=mask,
        n_points=int(mask.sum()),
    )


def path_to_time(
    path: AlignPath,
    query_ts: np.ndarray,
    source_ts: np.ndarray,
    query_start: float,
    query_end: float,
    cfg: AlignmentConfig,
    source_duration: float,
) -> tuple[float, float, FitResult]:
    """把对齐路径映射为原始素材的起止时间（秒）。"""
    q_times = query_ts[path.points[:, 0]]
    s_times = source_ts[path.points[:, 1]]
    fit = robust_time_fit(q_times, s_times, cfg)

    s_start = fit.slope * query_start + fit.intercept
    s_end = fit.slope * query_end + fit.intercept
    if s_end < s_start:
        s_start, s_end = s_end, s_start
    # 限制在素材时长内
    s_start = float(np.clip(s_start, 0.0, max(0.0, source_duration)))
    s_end = float(np.clip(s_end, 0.0, max(0.0, source_duration)))
    return s_start, s_end, fit


def compute_confidence(
    visual: float,
    coverage: float,
    margin: float,
    fit: FitResult,
    seg_len: float,
    cfg: ConfidenceConfig,
) -> tuple[float, dict]:
    """综合置信度：视觉相似 + 时序一致 + 来源区分度 + 对齐稳定性，并按片段长度衰减。"""
    visual_score = float(np.clip(visual, 0.0, 1.0))
    temporal_score = float(np.clip(coverage, 0.0, 1.0))
    margin_score = float(np.clip(margin / max(cfg.ambiguous_margin * 4.0, 1e-6), 0.0, 1.0))
    # 对齐稳定性：线性拟合优度 + 残差（秒）惩罚
    resid_score = float(np.exp(-max(fit.rmse, 0.0) / 0.5)) if np.isfinite(fit.rmse) else 0.0
    alignment_score = float(np.clip(0.5 * fit.r2 + 0.5 * resid_score, 0.0, 1.0))

    w = cfg
    total_w = w.w_visual + w.w_temporal + w.w_margin + w.w_alignment
    raw = (
        w.w_visual * visual_score
        + w.w_temporal * temporal_score
        + w.w_margin * margin_score
        + w.w_alignment * alignment_score
    ) / max(total_w, 1e-9)

    # 短片段（信息量少）适度衰减
    length_factor = float(np.clip(seg_len / max(cfg.length_ref_sec, 1e-6), 0.4, 1.0))
    length_factor = 0.85 + 0.15 * length_factor
    conf = float(np.clip(raw * length_factor, 0.0, 1.0))

    return conf, {
        "visual_score": visual_score,
        "temporal_score": temporal_score,
        "margin_score": margin_score,
        "alignment_score": alignment_score,
        "margin": float(margin),
        "fit_slope": fit.slope,
        "fit_rmse": fit.rmse if np.isfinite(fit.rmse) else -1.0,
        "fit_r2": fit.r2,
        "n_inliers": fit.n_points,
        "length_factor": length_factor,
    }


def decide_status(conf: float, margin: float, cfg: ConfidenceConfig) -> str:
    """置信度分级；歧义（多个候选分数接近）时降级，绝不硬给来源。"""
    if margin < cfg.ambiguous_margin and conf < cfg.high:
        return "UNKNOWN"
    if conf >= cfg.high:
        return "HIGH"
    if conf >= cfg.medium:
        return "MEDIUM"
    if conf >= cfg.low:
        return "LOW"
    return "UNKNOWN"


def merge_segments(segments: list[TimeSegment], max_gap: float = 0.35, max_time_jump: float = 0.6) -> list[TimeSegment]:
    """合并被镜头检测过度切分的相邻片段。

    合并条件：同一 source、成片时间相接、且原素材时间也相接（按各自 speed 推算）。
    """
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        same_source = (
            prev.source is not None
            and prev.source == seg.source
            and prev.status != "UNKNOWN"
            and seg.status != "UNKNOWN"
        )
        if same_source and abs(seg.query_start - prev.query_end) <= max_gap:
            assert prev.source_end is not None and seg.source_start is not None
            expect = prev.source_end + (seg.query_start - prev.query_end) * prev.speed
            if abs(seg.source_start - expect) <= max_time_jump and abs(seg.speed - prev.speed) < 0.12:
                prev.query_end = seg.query_end
                prev.source_end = seg.source_end
                w1 = prev.query_end - prev.query_start
                w2 = seg.query_end - seg.query_start
                prev.confidence = float((prev.confidence * w1 + seg.confidence * w2) / max(w1 + w2, 1e-9))
                prev.scores = {
                    k: float((prev.scores.get(k, 0.0) * w1 + seg.scores.get(k, 0.0) * w2) / max(w1 + w2, 1e-9))
                    for k in set(prev.scores) | set(seg.scores)
                }
                continue
        out.append(seg)
    for i, s in enumerate(out, 1):
        s.id = i
    return out
