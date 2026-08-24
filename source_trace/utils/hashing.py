"""文件指纹与缓存键计算。

大文件全量 SHA256 很慢，这里采用「分段采样 + 文件大小 + mtime」的快速指纹，
默认对首尾及中间共若干块做 SHA256，足以判断素材是否被替换/重新编码。
可通过 config.cache.full_hash=True 切换为全量 SHA256。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # 1MiB


def file_fingerprint(path: Path | str, full: bool = False) -> str:
    """返回文件内容指纹（十六进制字符串）。

    full=False 时读取首块、中块、尾块 + 文件大小；
    full=True 时读取全部内容做 SHA256。
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())

    with path.open("rb") as f:
        if full or size <= 3 * _CHUNK:
            while True:
                block = f.read(_CHUNK)
                if not block:
                    break
                h.update(block)
        else:
            offsets = [0, max(0, size // 2 - _CHUNK // 2), max(0, size - _CHUNK)]
            for off in offsets:
                f.seek(off)
                h.update(f.read(_CHUNK))
    return h.hexdigest()


def params_hash(params: dict[str, Any]) -> str:
    """对影响特征结果的参数做哈希，参数变化时缓存自动失效。"""
    blob = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
