"""设备检测：CUDA / CPU 自动选择，并打印显卡信息。"""

from __future__ import annotations

from dataclasses import dataclass

from .log import get_logger


@dataclass
class DeviceInfo:
    device: str  # "cuda" / "cpu"
    name: str
    total_vram_mb: float
    fp16: bool
    torch_version: str
    cuda_version: str | None

    def describe(self) -> str:
        if self.device == "cuda":
            return (
                f"device=cuda  GPU={self.name}  VRAM={self.total_vram_mb:.0f}MB  "
                f"FP16={'启用' if self.fp16 else '禁用'}  torch={self.torch_version}  CUDA={self.cuda_version}"
            )
        return f"device=cpu  CPU={self.name}  torch={self.torch_version}  （未检测到可用 CUDA，已回退 CPU）"


def resolve_device(prefer: str = "auto", allow_fp16: bool = True) -> DeviceInfo:
    """解析运行设备。prefer 可为 auto/cuda/cpu。"""
    log = get_logger()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise RuntimeError(
            "未安装 PyTorch，无法进行 ISC 特征提取。请执行：\n"
            "  python -m pip install torch==2.6.0 torchvision==0.21.0"
        ) from exc

    import platform

    cuda_ok = torch.cuda.is_available() and prefer in ("auto", "cuda")
    if prefer == "cuda" and not torch.cuda.is_available():
        log.warning("指定了 --device cuda，但当前环境 CUDA 不可用，自动回退 CPU")

    if cuda_ok:
        props = torch.cuda.get_device_properties(0)
        info = DeviceInfo(
            device="cuda",
            name=props.name,
            total_vram_mb=props.total_memory / 1024 / 1024,
            fp16=allow_fp16,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    else:
        info = DeviceInfo(
            device="cpu",
            name=platform.processor() or platform.machine(),
            total_vram_mb=0.0,
            fp16=False,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
    return info


def peak_vram_mb() -> float:
    """返回本进程 CUDA 峰值显存占用（MB）；CPU 环境返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
    except Exception:
        pass
    return 0.0
