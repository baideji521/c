"""特征缓存、维度/modality 兼容性校验测试。"""

from __future__ import annotations

import numpy as np
import pytest

from source_trace.config import FeatureConfig
from source_trace.features.cache import FeatureSet, FeatureStore
from source_trace.features.extractor import ColorHistExtractor, build_extractor
from source_trace.utils.hashing import file_fingerprint, params_hash
from source_trace.video.probe import probe


def _fs(dim=8, n=5, meta=None) -> FeatureSet:
    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, dim)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return FeatureSet(
        path="x.mp4",
        descriptors=d,
        timestamps=np.arange(n, dtype=np.float64),
        sample_fps=1.0,
        meta=meta or {"model": "m", "modality": "image", "dim": dim, "normalization": "l2", "weights": "none"},
    )


def test_slice_time():
    fs = _fs(n=10)
    sub = fs.slice_time(2.0, 5.0)
    assert len(sub) == 4
    assert sub.timestamps[0] == 2.0 and sub.timestamps[-1] == 5.0


def test_incompatible_modality_raises():
    """图像 embedding 与文本 embedding 绝不能混用。"""
    a = _fs(meta={"model": "m", "modality": "image", "dim": 8, "normalization": "l2", "weights": "none"})
    b = _fs(meta={"model": "m", "modality": "text", "dim": 8, "normalization": "l2", "weights": "none"})
    with pytest.raises(ValueError, match="modality"):
        a.assert_compatible(b)


def test_incompatible_dim_raises():
    a = _fs(dim=8)
    b = _fs(dim=16, meta={"model": "m", "modality": "image", "dim": 16, "normalization": "l2", "weights": "none"})
    with pytest.raises(ValueError, match="dim"):
        a.assert_compatible(b)


def test_incompatible_model_raises():
    a = _fs()
    b = _fs(meta={"model": "other", "modality": "image", "dim": 8, "normalization": "l2", "weights": "none"})
    with pytest.raises(ValueError, match="model"):
        a.assert_compatible(b)


def test_fingerprint_changes_with_content(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello" * 1000)
    fp1 = file_fingerprint(p)
    p.write_bytes(b"hellp" * 1000)
    fp2 = file_fingerprint(p)
    assert fp1 != fp2
    assert file_fingerprint(p) == fp2


def test_params_hash_stable():
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_cache_roundtrip_and_invalidation(tmp_path, source01):
    info = probe(source01)
    store = FeatureStore(tmp_path / "cache", enabled=True)
    fs = _fs(n=4)
    params = {"fps": 1.0}
    assert store.load(info, "isc", params) is None
    store.save(info, "isc", params, fs)
    got = store.load(info, "isc", params)
    assert got is not None
    np.testing.assert_allclose(got.descriptors, fs.descriptors, atol=1e-6)
    np.testing.assert_allclose(got.timestamps, fs.timestamps)
    # 参数变化 -> 缓存失效
    assert store.load(info, "isc", {"fps": 2.0}) is None


def test_cache_write_failure_is_not_fatal(tmp_path, source01, monkeypatch):
    """缓存写入失败（Windows 上 WinError 32 常见）只能降级，不能中断定位流程。"""
    from pathlib import Path as _P

    info = probe(source01)
    store = FeatureStore(tmp_path / "cache", enabled=True)
    orig = _P.replace

    def boom(self, target):  # noqa: ANN001, ARG001
        raise OSError(32, "另一个程序正在使用此文件")

    monkeypatch.setattr(_P, "replace", boom)
    store.save(info, "isc", {"fps": 1.0}, _fs(n=3))  # 不应抛异常
    monkeypatch.setattr(_P, "replace", orig)
    assert store.load(info, "isc", {"fps": 1.0}) is None
    assert not list((tmp_path / "cache").rglob("*.tmp")), "临时文件不应残留"


def test_cache_second_run_is_faster(tmp_path, source01):
    """第二次运行必须命中缓存（不重新提特征）。"""
    import time

    info = probe(source01)
    store = FeatureStore(tmp_path / "cache", enabled=True)
    ex = ColorHistExtractor()
    t0 = time.perf_counter()
    a = store.get_or_build(info, ex, fps=0.5, frame_size=64, tag="t")
    t1 = time.perf_counter()
    b = store.get_or_build(info, ex, fps=0.5, frame_size=64, tag="t")
    t2 = time.perf_counter()
    assert len(a) == len(b)
    np.testing.assert_allclose(a.descriptors, b.descriptors, atol=1e-6)
    assert (t2 - t1) < (t1 - t0) * 0.5


def test_colorhist_descriptor_properties():
    ex = ColorHistExtractor()
    frames = np.random.default_rng(0).integers(0, 256, size=(3, 32, 48, 3), dtype=np.uint8)
    d = ex.encode(frames)
    assert d.shape == (3, ex.dim)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)
    assert ex.meta["modality"] == "image" and ex.meta["normalization"] == "l2"


def test_build_extractor_colorhist_selection():
    ex = build_extractor(FeatureConfig(model="colorhist"))
    assert isinstance(ex, ColorHistExtractor)
