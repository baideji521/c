"""局部高密度精定位。

粗定位（1 FPS）只用来找到「哪个素材 + 大致区间」；
精定位只在该区间上以更高帧率（4~8 FPS）重新提特征并再做一次时序对齐，
因此既能把误差压到 ±0.2s 量级，又不会对整段视频做高帧率提特征。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SourceTraceConfig
from ..features.cache import FeatureSet, FeatureStore
from ..temporal.alignment import AlignPath, align
from ..temporal.mapping import FitResult, path_to_time, robust_time_fit
from ..temporal.similarity import build_sim_matrix, sample_background, score_normalize_with_desc
from ..utils.log import get_logger
from ..video.probe import VideoInfo


@dataclass
class RefineResult:
    source_start: float
    source_end: float
    visual: float
    coverage: float
    fit: FitResult
    path: AlignPath | None
    fps: float
    n_query_frames: int


def refine_segment(
    cfg: SourceTraceConfig,
    store: FeatureStore,
    extractor,
    query_info: VideoInfo,
    source_info: VideoInfo,
    query_start: float,
    query_end: float,
    coarse_source_start: float,
    coarse_source_end: float,
    fps: float,
    index=None,
    exclude_source_id: int | None = None,
) -> RefineResult | None:
    """在候选区间内做高帧率精定位。返回 None 表示精定位失败（保留粗定位结果）。"""
    log = get_logger()
    pad = cfg.sampling.fine_pad_sec
    s_lo = max(0.0, min(coarse_source_start, coarse_source_end) - pad)
    s_hi = min(source_info.duration, max(coarse_source_start, coarse_source_end) + pad)
    if s_hi - s_lo < 0.05:
        return None

    q_fs = store.build_segment(
        query_info, extractor, fps=fps, frame_size=cfg.sampling.frame_size, start=query_start, end=query_end
    )
    s_fs = store.build_segment(
        source_info, extractor, fps=fps, frame_size=cfg.sampling.frame_size, start=s_lo, end=s_hi
    )
    if len(q_fs) == 0 or len(s_fs) == 0:
        return None

    sim = build_sim_matrix(q_fs.descriptors, q_fs.timestamps, s_fs.descriptors, s_fs.timestamps)
    if cfg.refinement.score_norm and index is not None and exclude_source_id is not None:
        bg = sample_background(index, exclude_source_id, n=256)
        if bg.size:
            sim = score_normalize_with_desc(sim, q_fs.descriptors, bg, k=cfg.refinement.score_norm_k, beta=0.5)
            # 归一化后的分数用于建图，但最终 visual 分数仍取原始余弦相似度
            raw = build_sim_matrix(q_fs.descriptors, q_fs.timestamps, s_fs.descriptors, s_fs.timestamps)
        else:
            raw = sim
    else:
        raw = sim

    paths = align(sim.matrix, cfg.alignment, max_paths=3)
    if not paths:
        return None
    path = paths[0]

    s_start, s_end, fit = path_to_time(
        path,
        q_fs.timestamps,
        s_fs.timestamps,
        query_start,
        query_end,
        cfg.alignment,
        source_info.duration,
    )
    # visual 用原始余弦相似度重算（避免归一化后数值失真）
    vis = float(raw.matrix[path.points[:, 0], path.points[:, 1]].mean())
    log.debug(
        "精定位 %.2ffps：[%.2f,%.2f] -> [%.3f,%.3f] k=%.3f rmse=%.3f 点数=%d",
        fps, query_start, query_end, s_start, s_end, fit.slope, fit.rmse, path.n_points,
    )
    return RefineResult(
        source_start=s_start,
        source_end=s_end,
        visual=vis,
        coverage=path.coverage,
        fit=fit,
        path=path,
        fps=fps,
        n_query_frames=len(q_fs),
    )


def refit_from_points(
    q_times: np.ndarray, s_times: np.ndarray, cfg: SourceTraceConfig
) -> FitResult:
    """便于 benchmark / 单测直接复用的拟合入口。"""
    return robust_time_fit(q_times, s_times, cfg.alignment)


def slice_features(fs: FeatureSet, start: float, end: float) -> FeatureSet:
    return fs.slice_time(start, end)
