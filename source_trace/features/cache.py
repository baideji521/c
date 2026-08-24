"""特征缓存（增量处理）。

缓存键 = 文件内容指纹 + 影响特征的参数指纹（模型 / 采样率 / 分辨率 / TTA ...）。
源文件或参数任一变化，缓存自动失效并重算。

存储格式：``cache/<tag>/<file_fp>_<param_fp>.npz``
    descriptors : float32 (N, D)
    timestamps  : float64 (N,)
    meta        : json 字符串（模型、modality、维度、归一化、视频元信息）
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.hashing import file_fingerprint, params_hash
from ..utils.log import get_logger
from ..video.probe import VideoInfo
from ..video.reader import extract_frames


@dataclass
class FeatureSet:
    """一个视频（或区间）的帧级描述子集合。"""

    path: str
    descriptors: np.ndarray  # (N, D) float32, L2 归一化
    timestamps: np.ndarray  # (N,) float64 秒
    sample_fps: float
    meta: dict

    def __len__(self) -> int:
        return int(self.descriptors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.descriptors.shape[1]) if self.descriptors.size else 0

    def slice_time(self, start: float, end: float) -> "FeatureSet":
        m = (self.timestamps >= start - 1e-9) & (self.timestamps <= end + 1e-9)
        return FeatureSet(
            path=self.path,
            descriptors=self.descriptors[m],
            timestamps=self.timestamps[m],
            sample_fps=self.sample_fps,
            meta=self.meta,
        )

    def assert_compatible(self, other: "FeatureSet") -> None:
        """防止静默错误：不同 modality / 模型 / 维度的向量不得混用。"""
        for key in ("model", "modality", "dim", "normalization", "weights"):
            a, b = self.meta.get(key), other.meta.get(key)
            if a != b:
                raise ValueError(
                    f"特征不兼容，禁止混用：{key} 不一致（{a!r} vs {b!r}）。"
                    f"请清空 cache 后使用同一模型重新提取。"
                )


class FeatureStore:
    """特征缓存管理。"""

    def __init__(self, cache_dir: Path | str, enabled: bool = True, full_hash: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.full_hash = full_hash
        self._fp_cache: dict[str, str] = {}

    def fingerprint(self, path: Path | str) -> str:
        key = str(Path(path).resolve())
        if key not in self._fp_cache:
            self._fp_cache[key] = file_fingerprint(key, full=self.full_hash)
        return self._fp_cache[key]

    def _cache_path(self, info: VideoInfo, tag: str, params: dict) -> Path:
        fp = self.fingerprint(info.path)
        return self.cache_dir / tag / f"{fp[:24]}_{params_hash(params)}.npz"

    def load(self, info: VideoInfo, tag: str, params: dict) -> FeatureSet | None:
        if not self.enabled:
            return None
        p = self._cache_path(info, tag, params)
        if not p.exists():
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                return FeatureSet(
                    path=info.path,
                    descriptors=z["descriptors"].astype(np.float32),
                    timestamps=z["timestamps"].astype(np.float64),
                    sample_fps=float(meta.get("sample_fps", 1.0)),
                    meta=meta,
                )
        except Exception as exc:
            get_logger().warning("缓存读取失败，将重新提取（%s）：%s", type(exc).__name__, p.name)
            return None

    def save(self, info: VideoInfo, tag: str, params: dict, fs: FeatureSet) -> None:
        """写缓存。任何失败都只警告不抛出——缓存只是加速手段，不能影响定位结果。

        Windows 上 `Path.replace()` 可能因杀毒软件/索引服务短暂占用新建文件而抛
        WinError 32；另外内容完全相同的两个素材指纹一致、临时文件名也会撞在一起。
        因此临时文件名带上进程号与随机后缀，并对重命名做几次重试。
        """
        if not self.enabled:
            return
        log = get_logger()
        try:
            p = self._cache_path(info, tag, params)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            # 注意：np.savez_compressed 会给不以 .npz 结尾的路径自动追加后缀，
            # 因此这里传入已打开的文件对象，保证落盘路径可控。
            with tmp.open("wb") as f:
                np.savez_compressed(
                    f,
                    descriptors=fs.descriptors.astype(np.float32),
                    timestamps=fs.timestamps.astype(np.float64),
                    meta=json.dumps(fs.meta, ensure_ascii=False),
                )
            last: Exception | None = None
            for attempt in range(5):
                try:
                    tmp.replace(p)
                    return
                except OSError as exc:
                    last = exc
                    time.sleep(0.1 * (attempt + 1))
            tmp.unlink(missing_ok=True)
            log.warning("特征缓存写入失败（不影响本次结果）：%s（%s）", p.name, last)
        except Exception as exc:
            log.warning("特征缓存写入失败（不影响本次结果）：%s: %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------
    def get_or_build(
        self,
        info: VideoInfo,
        extractor,
        fps: float,
        frame_size: int,
        tag: str = "isc",
        chunk_sec: float = 90.0,
    ) -> FeatureSet:
        """取缓存，否则整段抽帧提特征并写缓存。"""
        log = get_logger()
        params = {
            "fps": round(float(fps), 6),
            "frame_size": int(frame_size),
            **{k: extractor.meta.get(k) for k in ("model", "weights", "modality", "dim", "normalization")},
            "tta": extractor.meta.get("tta"),
        }
        cached = self.load(info, tag, params)
        if cached is not None:
            log.info(
                "命中特征缓存：%s（%d 帧 @%.2ffps）", Path(info.path).name, len(cached), cached.sample_fps
            )
            return cached

        log.info("正在建立 %s 特征...（%.2ffps, %dpx）", Path(info.path).name, fps, frame_size)
        descs: list[np.ndarray] = []
        stamps: list[np.ndarray] = []
        t = 0.0
        while t < info.duration - 1e-6:
            end = min(info.duration, t + chunk_sec)
            batch = extract_frames(info, fps=fps, size=frame_size, start=t, end=end)
            if len(batch):
                descs.append(extractor.encode(batch.frames))
                stamps.append(batch.timestamps)
            t = end

        if not descs:
            raise RuntimeError(f"未能从 {Path(info.path).name} 提取任何帧特征")

        fs = FeatureSet(
            path=info.path,
            descriptors=np.concatenate(descs, axis=0),
            timestamps=np.concatenate(stamps, axis=0),
            sample_fps=float(fps),
            meta={
                **extractor.meta,
                "sample_fps": float(fps),
                "frame_size": int(frame_size),
                "video": {
                    "name": Path(info.path).name,
                    "duration": info.duration,
                    "fps": info.fps,
                    "width": info.width,
                    "height": info.height,
                    "codec": info.codec,
                    "is_vfr": info.is_vfr,
                },
            },
        )
        self.save(info, tag, params, fs)
        log.info("特征提取完成：%s -> %d 帧 x %dD", Path(info.path).name, len(fs), fs.dim)
        return fs

    def build_segment(
        self,
        info: VideoInfo,
        extractor,
        fps: float,
        frame_size: int,
        start: float,
        end: float,
    ) -> FeatureSet:
        """按区间提取特征（精定位用，不落盘）。"""
        batch = extract_frames(info, fps=fps, size=frame_size, start=start, end=end)
        if len(batch) == 0:
            return FeatureSet(
                path=info.path,
                descriptors=np.zeros((0, getattr(extractor, "dim", 0)), dtype=np.float32),
                timestamps=np.zeros((0,), dtype=np.float64),
                sample_fps=fps,
                meta={**extractor.meta, "sample_fps": fps, "frame_size": frame_size},
            )
        return FeatureSet(
            path=info.path,
            descriptors=extractor.encode(batch.frames),
            timestamps=batch.timestamps,
            sample_fps=float(fps),
            meta={**extractor.meta, "sample_fps": float(fps), "frame_size": int(frame_size)},
        )
