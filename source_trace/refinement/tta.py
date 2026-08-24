"""TTA（Test-Time Augmentation）与分数归一化的封装。

实际的多尺度 / 水平翻转推理在 ``features.extractor.ISCExtractor`` 内实现，
本模块只提供开关封装与说明，供困难样本流程调用，避免第一阶段就全量开启。
"""

from __future__ import annotations

from contextlib import contextmanager

from ..features.extractor import TTASpec

__all__ = ["TTASpec", "tta_enabled"]


@contextmanager
def tta_enabled(extractor, enabled: bool = True):
    """临时开启/关闭 extractor 的 TTA。"""
    setter = getattr(extractor, "set_tta", None)
    if setter is None:
        yield extractor
        return
    prev = bool(getattr(extractor, "tta", TTASpec()).enabled)
    setter(enabled)
    try:
        yield extractor
    finally:
        setter(prev)
