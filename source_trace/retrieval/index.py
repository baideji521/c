"""多素材帧级索引。

每个 source 独立保存特征（绝不把多个视频的时间轴 concat 混淆），
但为了高效检索，向量在内存中拼成一张大矩阵，并同时保存
``owner`` （所属 source 序号）与 ``local_index`` （在该 source 内的帧序号）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..features.cache import FeatureSet
from ..utils.log import get_logger


@dataclass
class SourceEntry:
    source_id: int
    name: str
    path: str
    features: FeatureSet
    # 内容完全相同的其他素材文件名（同一份素材被复制成多份时）
    duplicates: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.features.meta.get("video", {}).get("duration", 0.0))

    @property
    def display_name(self) -> str:
        if not self.duplicates:
            return self.name
        return f"{self.name}（内容与 {'、'.join(self.duplicates)} 相同）"


class SourceIndex:
    """所有原始素材的帧级检索索引。"""

    def __init__(self, backend: str = "auto") -> None:
        self.entries: list[SourceEntry] = []
        self._matrix: np.ndarray | None = None
        self._owner: np.ndarray | None = None
        self._local: np.ndarray | None = None
        self._faiss = None
        self.backend = backend

    # ------------------------------------------------------------------
    def add(self, name: str, features: FeatureSet) -> SourceEntry:
        if self.entries:
            self.entries[0].features.assert_compatible(features)
        entry = SourceEntry(source_id=len(self.entries) + 1, name=name, path=features.path, features=features)
        self.entries.append(entry)
        self._matrix = None
        return entry

    def by_name(self, name: str) -> SourceEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def get(self, source_id: int) -> SourceEntry:
        return self.entries[source_id - 1]

    @property
    def size(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    def build(self) -> None:
        """拼接向量矩阵并（可选）建立 faiss 索引。"""
        log = get_logger()
        if not self.entries:
            raise RuntimeError("索引为空：没有任何可用的原始素材")
        mats, owners, locals_ = [], [], []
        for e in self.entries:
            d = e.features.descriptors
            mats.append(d)
            owners.append(np.full(d.shape[0], e.source_id, dtype=np.int32))
            locals_.append(np.arange(d.shape[0], dtype=np.int32))
        self._matrix = np.ascontiguousarray(np.concatenate(mats, axis=0), dtype=np.float32)
        self._owner = np.concatenate(owners)
        self._local = np.concatenate(locals_)

        backend = self.backend
        if backend == "auto":
            backend = "faiss" if self._matrix.shape[0] > 20000 and _has_faiss() else "numpy"
        if backend == "faiss":
            try:
                import faiss

                index = faiss.IndexFlatIP(self._matrix.shape[1])
                index.add(self._matrix)
                self._faiss = index
            except Exception as exc:
                log.warning("faiss 初始化失败（%s），使用 numpy 检索", exc)
                self._faiss = None
                backend = "numpy"
        log.info(
            "索引就绪：%d 个素材，%d 帧 x %dD，后端=%s",
            self.size, self._matrix.shape[0], self._matrix.shape[1], backend,
        )

    # ------------------------------------------------------------------
    def search(self, queries: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """帧级检索。

        返回 (sims, owner_ids, local_idx)，形状均为 (Q, top_n)。
        """
        if self._matrix is None:
            self.build()
        assert self._matrix is not None and self._owner is not None and self._local is not None
        top_n = int(min(top_n, self._matrix.shape[0]))
        if queries.shape[0] == 0:
            z = np.zeros((0, top_n))
            return z.astype(np.float32), z.astype(np.int32), z.astype(np.int32)

        if self._faiss is not None:
            sims, idx = self._faiss.search(np.ascontiguousarray(queries, dtype=np.float32), top_n)
        else:
            sim_all = queries.astype(np.float32) @ self._matrix.T
            idx = np.argpartition(-sim_all, top_n - 1, axis=1)[:, :top_n]
            sims = np.take_along_axis(sim_all, idx, axis=1)
            order = np.argsort(-sims, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            sims = np.take_along_axis(sims, order, axis=1)
        return sims.astype(np.float32), self._owner[idx], self._local[idx]

    def similarity_matrix(self, query_desc: np.ndarray, source_id: int) -> np.ndarray:
        """query 与指定 source 的完整帧-帧余弦相似度矩阵 (Q, S)。"""
        e = self.get(source_id)
        return query_desc.astype(np.float32) @ e.features.descriptors.astype(np.float32).T


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401

        return True
    except Exception:
        return False


def build_index_from_dir(
    sources: list[Path],
    store,
    extractor,
    fps: float,
    frame_size: int,
    backend: str = "auto",
) -> tuple[SourceIndex, list[dict]]:
    """扫描并为每个素材建立特征；单个素材出错不影响其他素材。

    返回 (index, errors)。
    """
    from ..video.probe import probe

    log = get_logger()
    index = SourceIndex(backend=backend)
    errors: list[dict] = []
    # 内容完全相同的素材（同一份文件被复制成多份）只入索引一次：
    # 否则两份一模一样的素材会互相拉平 margin，被判成「歧义」而输出 UNKNOWN。
    seen: dict[str, SourceEntry] = {}
    for p in sources:
        try:
            info = probe(p)
            fp = store.fingerprint(p) if hasattr(store, "fingerprint") else None
            if fp is not None and fp in seen:
                dup = seen[fp]
                dup.duplicates.append(p.name)
                log.info("素材 %s 与 %s 内容完全相同，只保留一份参与检索", p.name, dup.name)
                continue
            fs = store.get_or_build(info, extractor, fps=fps, frame_size=frame_size, tag="isc")
            entry = index.add(p.name, fs)
            if fp is not None:
                seen[fp] = entry
        except Exception as exc:
            log.error("素材处理失败，已跳过：%s（%s: %s）", p.name, type(exc).__name__, exc)
            errors.append({"source": p.name, "path": str(p), "error": f"{type(exc).__name__}: {exc}"})
    if index.size == 0:
        raise RuntimeError("所有原始素材都处理失败，无法继续")
    index.build()
    return index, errors
