"""ISC21 descriptor 模型定义与权重加载。

结构对齐 lyakaap/ISC21-Descriptor-Track-1st（ISC 2021 Descriptor Track 第一名）：

    timm backbone(features_only) -> GeM pooling -> Linear(fc_dim) -> BatchNorm1d -> L2 normalize

训练时 GeM 的 p 为可学习/固定值，推理时使用 eval_p（原实现为 1.0，即等价 average pooling）。
本项目只做 inference，不训练。

权重缺失时的降级策略（不阻塞主流程）：
    ISC 权重 -> ImageNet 预训练 backbone + GeM + L2（无 fc/bn 投影）
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.log import get_logger


def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
    """Generalized Mean Pooling。x: (B, C, H, W) -> (B, C, 1, 1)

    注意：``x**p``（p=4）在 FP16 下极易溢出为 inf，进而在 BatchNorm 处产生 NaN。
    因此幂运算强制在 FP32 中完成，再转回输入 dtype。
    """
    xf = x.float().clamp(min=eps).pow(p)
    out = torch.nn.functional.avg_pool2d(xf, (x.size(-2), x.size(-1))).pow(1.0 / p)
    return out.to(x.dtype)


class ISCNet(nn.Module):
    """ISC21 描述子网络。

    ``eval_p=4.0`` 与 ``fc`` 无 bias 均取自官方 checkpoint 中保存的训练参数
    （``gem_eval_p=4.0``、``module.fc.weight`` 形状 (256, 512) 且无 ``fc.bias``）。
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int,
        fc_dim: int = 256,
        p: float = 1.0,
        eval_p: float = 4.0,
        projection: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection
        self.p = p
        self.eval_p = eval_p
        if projection:
            self.fc = nn.Linear(in_channels, fc_dim, bias=False)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.out_dim = fc_dim
        else:
            self.fc = None
            self.bn = None
            self.out_dim = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f = feats[-1] if isinstance(feats, (list, tuple)) else feats
        p = self.p if self.training else self.eval_p
        v = gem(f, p).flatten(1)
        if self.fc is not None:
            v = self.fc(v)
            v = self.bn(v)
        return F.normalize(v, dim=1)


def _clean_state_dict(sd: dict) -> dict:
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    elif "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    out = {}
    for k, v in sd.items():
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        out[k] = v
    return out


def build_isc_model(
    backbone_name: str,
    fc_dim: int = 256,
    weights: Path | str | None = None,
    fallback_backbone: str | None = None,
) -> tuple[ISCNet, dict]:
    """构建模型并尽力加载权重。

    返回 (model, meta)。meta 记录 model/modality/dim/normalization/weights 状态，
    用于缓存校验与防止不同 embedding 空间混用。
    """
    log = get_logger()
    import timm

    has_weights = weights is not None and Path(weights).exists()
    # 有 ISC 权重时无需下载 ImageNet 预训练；否则必须依赖 ImageNet 权重
    pretrained = not has_weights

    def _create(name: str, pretrained_flag: bool):
        return timm.create_model(name, features_only=True, pretrained=pretrained_flag)

    used_backbone = backbone_name
    try:
        backbone = _create(backbone_name, pretrained)
    except Exception as exc:
        if not fallback_backbone:
            raise
        log.warning("backbone %s 创建失败（%s），改用 %s", backbone_name, exc, fallback_backbone)
        used_backbone = fallback_backbone
        backbone = _create(fallback_backbone, pretrained)

    in_channels = backbone.feature_info.channels()[-1]

    if has_weights:
        model = ISCNet(backbone, in_channels=in_channels, fc_dim=fc_dim, projection=True)
        sd = _clean_state_dict(torch.load(str(weights), map_location="cpu", weights_only=False))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        critical = [k for k in missing if k.startswith(("fc.", "bn."))]
        if critical:
            log.warning(
                "ISC 权重缺少投影层参数 %s，降级为 ImageNet backbone + GeM 特征", critical[:4]
            )
            backbone = _create(used_backbone, True)
            model = ISCNet(backbone, in_channels=in_channels, projection=False)
            weights_state = "fallback_imagenet"
        else:
            log.info(
                "已加载 ISC 权重：%s（missing=%d unexpected=%d）",
                Path(weights).name, len(missing), len(unexpected),
            )
            weights_state = f"isc:{Path(weights).name}"
    else:
        log.warning("未找到 ISC 预训练权重，使用 ImageNet 预训练 backbone + GeM（精度略降，主流程不受阻）")
        model = ISCNet(backbone, in_channels=in_channels, projection=False)
        weights_state = "fallback_imagenet"

    model.eval()
    meta = {
        "model": f"ISCNet/{used_backbone}",
        "weights": weights_state,
        "modality": "image",
        "dim": int(model.out_dim),
        "normalization": "l2",
        "pooling": "gem",
    }
    return model, meta
