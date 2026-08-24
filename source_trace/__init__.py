"""成片原素材反溯工具 / Source Trace.

从一个混剪成片（query）反向定位其每个片段来自哪个原始素材（source）以及在
原始素材中的准确起止时间，并自动裁剪导出。

核心链路：
    Shot Detection -> ISC21 descriptor -> Multi-Source Top-K Retrieval
    -> VCSL/TN Temporal Alignment -> Local Fine Match -> FFmpeg Export
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
