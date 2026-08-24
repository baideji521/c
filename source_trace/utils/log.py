"""中文日志系统。

统一格式：[INFO] 消息
支持写入文件，便于每个阶段保存日志。
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_LOGGER_NAME = "source_trace"
_CONFIGURED = False


class _PlainFormatter(logging.Formatter):
    """输出形如 ``[INFO] 扫描原始素材...`` 的中文日志。"""

    _LEVEL = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: "WARN",
        logging.ERROR: "ERROR",
        logging.CRITICAL: "FATAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        level = self._LEVEL.get(record.levelno, record.levelname)
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        msg = record.getMessage()
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"
        return f"{ts} [{level}] {msg}"


def setup_logging(level: int = logging.INFO, log_file: Path | str | None = None) -> logging.Logger:
    """初始化全局 logger。重复调用只会追加文件 handler。"""
    global _CONFIGURED
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not _CONFIGURED:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(_PlainFormatter())
        logger.addHandler(stream)
        _CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            getattr(h, "baseFilename", None) for h in logger.handlers if isinstance(h, logging.FileHandler)
        }
        if str(log_file.resolve()) not in existing:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(_PlainFormatter())
            logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    """获取全局 logger（未初始化时自动初始化）。"""
    if not _CONFIGURED:
        return setup_logging()
    return logging.getLogger(_LOGGER_NAME)


class Timer:
    """累计各阶段耗时，供 PERFORMANCE_REPORT 使用。"""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.stages[name] = self.stages.get(name, 0.0) + dt

    def add(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + seconds

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float]:
        d = {k: round(v, 3) for k, v in self.stages.items()}
        d["total"] = round(self.total, 3)
        return d
