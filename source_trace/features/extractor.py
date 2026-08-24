"""帧描述子提取器。

对外统一接口：``encode(frames) -> (N, D) float32``，输入 uint8 RGB (N,H,W,3)，
输出 L2 归一化向量。

两种实现：
* ``ISCExtractor``   —— ISC21 网络（PyTorch，GPU/CPU 自动），主用
* ``ColorHistExtractor`` —— 纯 numpy 颜色/梯度直方图，作为 torch 缺失时的兜底与 benchmark 基线

所有 extractor 都必须暴露 ``meta``（model/modality/dim/normalization），
缓存与检索前会校验，严禁不同 modality 或不同模型的向量混用。
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..config import FeatureConfig
from ..utils.device import DeviceInfo, resolve_device
from ..utils.log import get_logger

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Descriptor(Protocol):
    meta: dict
    dim: int

    def encode(self, frames: np.ndarray) -> np.ndarray: ...


@dataclass
class TTASpec:
    enabled: bool = False
    scales: tuple[float, ...] = (1.0,)
    hflip: bool = False


class ColorHistExtractor:
    """纯 numpy 兜底描述子：分块颜色直方图 + 梯度方向直方图。

    不依赖 torch，可在任何环境运行；鲁棒性弱于 ISC，仅作 fallback / baseline。
    """

    def __init__(self, grid: int = 4, color_bins: int = 8, grad_bins: int = 8) -> None:
        self.grid = grid
        self.color_bins = color_bins
        self.grad_bins = grad_bins
        self.dim = grid * grid * (3 * color_bins + grad_bins)
        self.meta = {
            "model": f"colorhist/g{grid}c{color_bins}o{grad_bins}",
            "weights": "none",
            "modality": "image",
            "dim": self.dim,
            "normalization": "l2",
            "pooling": "grid-hist",
        }
        self.center_crop = 1.0

    def set_center_crop(self, ratio: float) -> None:
        self.center_crop = float(max(0.1, min(1.0, ratio)))
        self.meta["center_crop"] = self.center_crop

    def encode(self, frames: np.ndarray) -> np.ndarray:
        if self.center_crop < 1.0 and frames.shape[0]:
            import cv2

            h, w = frames.shape[1], frames.shape[2]
            ch, cw = max(8, int(h * self.center_crop)), max(8, int(w * self.center_crop))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            frames = np.stack(
                [cv2.resize(f[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_AREA) for f in frames]
            )
        n, h, w, _ = frames.shape
        g = self.grid
        out = np.zeros((n, self.dim), dtype=np.float32)
        ys = np.linspace(0, h, g + 1).astype(int)
        xs = np.linspace(0, w, g + 1).astype(int)
        f = frames.astype(np.float32)
        gray = f.mean(axis=3)
        gy = np.gradient(gray, axis=1)
        gx = np.gradient(gray, axis=2)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # 0~1

        for i in range(n):
            pos = 0
            for a in range(g):
                for b in range(g):
                    cell = f[i, ys[a] : ys[a + 1], xs[b] : xs[b + 1], :]
                    for c in range(3):
                        hist, _ = np.histogram(cell[..., c], bins=self.color_bins, range=(0, 256))
                        out[i, pos : pos + self.color_bins] = hist
                        pos += self.color_bins
                    m = mag[i, ys[a] : ys[a + 1], xs[b] : xs[b + 1]]
                    o = ang[i, ys[a] : ys[a + 1], xs[b] : xs[b + 1]]
                    hist, _ = np.histogram(o, bins=self.grad_bins, range=(0, 1), weights=m)
                    out[i, pos : pos + self.grad_bins] = hist
                    pos += self.grad_bins
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-8)

    def set_tta(self, enabled: bool) -> None:  # noqa: ARG002 - 兜底实现不支持 TTA
        return None


class ISCExtractor:
    """ISC21 描述子提取器（batch inference + FP16 + GPU/CPU 自动）。"""

    def __init__(self, cfg: FeatureConfig, tta: TTASpec | None = None) -> None:
        import torch

        self.cfg = cfg
        self.torch = torch
        self.tta = tta or TTASpec()
        self.device_info: DeviceInfo = resolve_device(cfg.device, cfg.fp16)
        get_logger().info("运行设备：%s", self.device_info.describe())

        weights = _resolve_weights(cfg)
        from .isc import build_isc_model

        model, meta = build_isc_model(
            cfg.backbone, fc_dim=cfg.dim, weights=weights, fallback_backbone=cfg.fallback_backbone
        )
        self.model = model.to(self.device_info.device)
        self.use_fp16 = bool(self.device_info.device == "cuda" and cfg.fp16)
        if self.use_fp16:
            self.model = self.model.half()
        self.meta = dict(meta)
        self.meta["device"] = self.device_info.device
        self.meta["fp16"] = self.use_fp16
        self.meta["tta"] = {"enabled": self.tta.enabled, "scales": list(self.tta.scales), "hflip": self.tta.hflip}
        self.center_crop = 1.0
        self.dim = int(meta["dim"])

    def set_center_crop(self, ratio: float) -> None:
        """中心裁切补偿：成片被裁边后，把原素材也裁到相同视野再比对。

        ratio=1.0 表示不裁切。裁切后会缩放回原尺寸，保证与未裁切分支的输入尺度一致。
        """
        self.center_crop = float(max(0.1, min(1.0, ratio)))
        self.meta["center_crop"] = self.center_crop

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = torch.from_numpy(batch).to(self.device_info.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.as_tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.as_tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        if self.center_crop < 1.0:
            h, w = x.shape[2], x.shape[3]
            ch, cw = max(8, int(h * self.center_crop)), max(8, int(w * self.center_crop))
            y0, x0 = (h - ch) // 2, (w - cw) // 2
            x = torch.nn.functional.interpolate(
                x[:, :, y0 : y0 + ch, x0 : x0 + cw], size=(h, w), mode="bilinear", align_corners=False
            )
        if self.use_fp16:
            x = x.half()

        views = [x]
        if self.tta.enabled:
            for s in self.tta.scales:
                if abs(s - 1.0) > 1e-3:
                    h = max(32, int(round(x.shape[2] * s / 2) * 2))
                    w = max(32, int(round(x.shape[3] * s / 2) * 2))
                    views.append(torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False))
            if self.tta.hflip:
                views = views + [torch.flip(v, dims=[3]) for v in list(views)]

        acc = None
        with torch.no_grad():
            for v in views:
                out = self.model(v).float()
                acc = out if acc is None else acc + out
        acc = acc / len(views)
        acc = torch.nn.functional.normalize(acc, dim=1)
        out = acc.cpu().numpy().astype(np.float32)
        if not np.isfinite(out).all():
            # FP16 数值溢出等原因导致 NaN/Inf：立即降级为 FP32 重算，绝不把脏向量传下去
            get_logger().error("检测到非有限特征值（FP16 溢出），本批改用 FP32 重算")
            self.use_fp16 = False
            self.model = self.model.float()
            self.meta["fp16"] = False
            return self._forward(batch)
        return out

    def encode(self, frames: np.ndarray) -> np.ndarray:
        if frames.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        bs = max(1, int(self.cfg.batch_size))
        outs = []
        for i in range(0, frames.shape[0], bs):
            outs.append(self._forward(np.ascontiguousarray(frames[i : i + bs])))
        return np.concatenate(outs, axis=0)

    def set_tta(self, enabled: bool) -> None:
        """困难样本时临时开启 TTA（多尺度 + 水平翻转）。"""
        self.tta.enabled = bool(enabled)
        self.meta["tta"] = {
            "enabled": self.tta.enabled,
            "scales": list(self.tta.scales),
            "hflip": self.tta.hflip,
        }


def _resolve_weights(cfg: FeatureConfig) -> Path | None:
    """定位 ISC 权重文件，必要时尝试下载（失败不阻塞）。"""
    log = get_logger()
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    if cfg.weights:
        p = Path(cfg.weights)
        if not p.is_absolute():
            p = models_dir / cfg.weights
        if p.exists():
            return p
        log.warning("指定的权重不存在：%s", p)

    for pattern in ("isc_ft_v107*", "*isc*.pth*", "*isc*.tar"):
        found = sorted(models_dir.glob(pattern))
        if found:
            return found[0]

    if not cfg.weight_urls:
        return None
    models_dir.mkdir(parents=True, exist_ok=True)
    for url in cfg.weight_urls:
        dest = models_dir / url.rsplit("/", 1)[-1]
        try:
            log.info("尝试下载 ISC 权重：%s", url)
            urllib.request.urlretrieve(url, dest)  # noqa: S310 - URL 来自配置
            if dest.stat().st_size > 1024 * 1024:
                log.info("权重下载完成：%s（%.1f MB）", dest.name, dest.stat().st_size / 1024 / 1024)
                return dest
            dest.unlink(missing_ok=True)
        except Exception as exc:
            log.warning("权重下载失败（%s）：%s", type(exc).__name__, exc)
            dest.unlink(missing_ok=True)
    return None


def build_extractor(cfg: FeatureConfig, tta: TTASpec | None = None) -> Descriptor:
    """按配置构建描述子提取器；torch 不可用时回退到 numpy 实现。"""
    log = get_logger()
    if cfg.model == "colorhist":
        return ColorHistExtractor()
    try:
        return ISCExtractor(cfg, tta=tta)
    except Exception as exc:
        log.error("ISC 提取器初始化失败（%s: %s），回退纯 numpy 描述子", type(exc).__name__, exc)
        return ColorHistExtractor()
