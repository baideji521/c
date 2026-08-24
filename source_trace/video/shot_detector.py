"""镜头检测（Shot Detection）。

成片不能按固定时间切分，必须按镜头边界切分，否则一个 shot 会横跨两个来源素材。

三种实现，自动降级：
1. ``pyscenedetect``：可选依赖，ContentDetector，效果最好
2. ``histogram``：内置实现，灰度直方图 + 像素差分双指标，无额外依赖
3. ``fixed``：固定窗口，兜底

输出 ``[{"shot_id": 1, "start": 0.0, "end": 4.2}, ...]``，时间单位为秒。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ShotConfig
from ..utils.log import get_logger
from .probe import VideoInfo
from .reader import extract_frames

_DETECT_FPS = 12.0
_DETECT_SIZE = 64


@dataclass
class Shot:
    shot_id: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0

    def to_dict(self) -> dict:
        return {"shot_id": self.shot_id, "start": round(self.start, 4), "end": round(self.end, 4)}


def detect_shots(info: VideoInfo, cfg: ShotConfig) -> list[Shot]:
    """检测镜头边界。"""
    log = get_logger()
    method = cfg.method

    if method == "auto":
        method = "pyscenedetect" if _has_pyscenedetect() else "histogram"

    cuts: list[float] = []
    if method == "pyscenedetect":
        try:
            cuts = _detect_pyscenedetect(info, cfg)
        except Exception as exc:
            log.warning("PySceneDetect 检测失败（%s），回退直方图方法", exc)
            method = "histogram"
    if method == "histogram":
        cuts = _detect_histogram(info, cfg)
    elif method == "fixed":
        cuts = list(np.arange(cfg.fixed_window_sec, info.duration, cfg.fixed_window_sec))

    shots = _build_shots(cuts, info.duration, cfg)
    log.info("镜头检测(%s)：%s -> %d 个镜头", method, Path(info.path).name, len(shots))
    return shots


def _has_pyscenedetect() -> bool:
    try:
        import scenedetect  # noqa: F401

        return True
    except Exception:
        return False


def _detect_pyscenedetect(info: VideoInfo, cfg: ShotConfig) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(info.path))
    sm = SceneManager()
    sm.add_detector(
        ContentDetector(
            threshold=cfg.content_threshold,
            min_scene_len=max(1, int(round(cfg.min_shot_sec * max(info.fps, 1.0)))),
        )
    )
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    # scenedetect 0.6.4 起 FrameTimecode.get_seconds() 已废弃，改用 .seconds 属性
    def _sec(tc) -> float:
        s = getattr(tc, "seconds", None)
        return float(s) if s is not None else float(tc.get_seconds())

    return [_sec(s[0]) for s in scenes[1:]]


def _detect_histogram(info: VideoInfo, cfg: ShotConfig) -> list[float]:
    """灰度直方图相关性 + 平均绝对差分，双指标投票。"""
    batch = extract_frames(info, fps=_DETECT_FPS, size=_DETECT_SIZE, gray=True)
    if len(batch) < 3:
        return []

    frames = batch.frames[:, :, :, 0].astype(np.float32)
    ts = batch.timestamps

    # 直方图（32 bin）逐帧
    n = frames.shape[0]
    hists = np.zeros((n, 32), dtype=np.float32)
    for i in range(n):
        h, _ = np.histogram(frames[i], bins=32, range=(0, 256))
        hists[i] = h
    hists /= np.maximum(hists.sum(axis=1, keepdims=True), 1e-6)

    # 直方图交集距离：1 - sum(min(h1,h2))
    hist_dist = 1.0 - np.minimum(hists[:-1], hists[1:]).sum(axis=1)
    # 像素平均绝对差（归一化到 0~1）
    pix_dist = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2)) / 255.0

    score = 0.5 * hist_dist + 0.5 * np.minimum(pix_dist * 3.0, 1.0)

    # 阈值：以「稳健统计量（中位数 + z·MAD）」为主，hist_threshold 只作为上限。
    # 不能把 hist_threshold 当下限：低对比素材的真实切点得分可能只有 0.3，
    # 用固定下限 0.35 会把所有切点全部漏掉（整段被当成一个镜头）。
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med)))
    adaptive = med + cfg.hist_z * 1.4826 * mad
    thr = min(cfg.hist_threshold, max(cfg.hist_min_abs, adaptive))

    cuts: list[float] = []
    last_cut = ts[0]
    for i, s in enumerate(score):
        if s < thr or s < cfg.hist_peak_ratio * med:
            continue
        # 只取局部极大值，避免一次切换连续触发多帧
        if i > 0 and score[i - 1] > s:
            continue
        if i + 1 < len(score) and score[i + 1] > s:
            continue
        t = float(ts[i + 1])
        if t - last_cut < cfg.min_shot_sec:
            continue
        cuts.append(t)
        last_cut = t
    return cuts


def _build_shots(cuts: list[float], duration: float, cfg: ShotConfig) -> list[Shot]:
    """由切点构造 shot 列表，并处理过短合并 / 过长切分。"""
    bounds = [0.0]
    for c in sorted(cuts):
        if c - bounds[-1] >= cfg.min_shot_sec and c < duration - 1e-3:
            bounds.append(float(c))
    bounds.append(float(duration))

    # 过长镜头均分切开（避免跨来源）
    refined: list[float] = [bounds[0]]
    for i in range(1, len(bounds)):
        prev, cur = refined[-1], bounds[i]
        length = cur - prev
        if cfg.max_shot_sec > 0 and length > cfg.max_shot_sec:
            parts = int(np.ceil(length / cfg.max_shot_sec))
            step = length / parts
            for k in range(1, parts):
                refined.append(prev + k * step)
        refined.append(cur)

    shots: list[Shot] = []
    for i in range(len(refined) - 1):
        start, end = refined[i], refined[i + 1]
        if end - start < 1e-3:
            continue
        shots.append(Shot(shot_id=len(shots) + 1, start=float(start), end=float(end)))

    # 最后一个镜头过短则并入前一个
    if len(shots) >= 2 and shots[-1].duration < cfg.min_shot_sec * 0.5:
        shots[-2].end = shots[-1].end
        shots.pop()
        for i, s in enumerate(shots):
            s.shot_id = i + 1
    return shots
