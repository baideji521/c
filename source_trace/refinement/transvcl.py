"""TransVCL 困难样本判别（可选依赖）。

定位：**不是**第一层算法。只有当 ISC + VCSL/TN 无法稳定确定来源
（confidence < enhance_threshold 且候选歧义）时才调用。

TransVCL 需要额外的模型权重与运行环境，官方实现依赖较重。
本模块做成完全可选：
* 未安装 / 无权重  -> ``is_available()`` 返回 False，主流程照常运行
* 已安装          -> 由 ``resolve()`` 给出候选排序的重打分

这样「没有 TransVCL 主程序依然能跑」这一硬性要求得到保证。
"""

from __future__ import annotations

from pathlib import Path

from ..utils.log import get_logger

_WEIGHT_NAMES = ("transvcl_model.pth", "transvcl.pth", "transVCL.pth")


def weights_path() -> Path | None:
    models = Path(__file__).resolve().parent.parent.parent / "models" / "transvcl"
    if not models.is_dir():
        return None
    for name in _WEIGHT_NAMES:
        p = models / name
        if p.exists():
            return p
    found = sorted(models.glob("*.pth"))
    return found[0] if found else None


def is_available() -> bool:
    """TransVCL 是否可用（同时要求包与权重存在）。"""
    if weights_path() is None:
        return False
    try:
        import importlib

        importlib.import_module("transvcl")
        return True
    except Exception:
        return False


def unavailable_reason() -> str:
    if weights_path() is None:
        return "缺少 TransVCL 权重（放置于 models/transvcl/*.pth）"
    return "未安装 transvcl 包（见 requirements-optional.txt）"


def resolve(*args, **kwargs):  # pragma: no cover - 需要可选依赖
    """对困难样本重新判别；不可用时返回 None，调用方需保留原结果。"""
    if not is_available():
        get_logger().info("TransVCL 不可用（%s），跳过困难样本重判别", unavailable_reason())
        return None
    raise NotImplementedError(
        "检测到 TransVCL 权重与依赖，但本项目尚未接入其推理适配层。"
        "请在此实现 features -> TransVCL -> 片段置信度 的调用。"
    )
