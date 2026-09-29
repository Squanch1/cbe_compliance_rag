"""原始文件的哈希计算。

用于判重：同一份文件重复导入时，靠文件内容的哈希识别出来，
不会产生第二条记录（见 docs/spec/03-data-model.md 3.2）。

**算的是文件字节，不是解析后的正文。** 这样判重发生在解析之前，
重复文件省下一次完整解析。代价是同一份内容的 PDF 版与 HTML 版
会被当作两份独立文档——它们的来源与解析结构本就不同，引用时
需要区分，因此这样处理是对的。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 分块读取的大小。取 1MiB：语料里有 105 页的 PDF，
# 一次性读进内存白占空间，而哈希本来就是流式的。
_CHUNK_SIZE = 1 << 20


class HashingError(Exception):
    """哈希计算无法继续时抛出。"""


def file_content_hash(path: Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """计算文件的 SHA-256，返回 64 个十六进制字符。

    **先判绝对路径再判存在性**：相对路径先报「文件不存在」会把
    排查方向带偏（见 CLAUDE.md 5.2）。

    按块读取而不是一次性读进内存。chunk_size 做成参数是为了让测试
    能造出跨块的文件，验证边界处不漏字节也不重复读；生产上它也是
    真实的调优点，大文件场景可以调大。
    """
    if not path.is_absolute():
        raise HashingError("path 必须是绝对路径，收到：%s" % path)
    if chunk_size <= 0:
        # read(0) 永远返回空，循环立刻结束，会静默算出空文件的哈希
        raise HashingError("chunk_size 必须为正数，收到：%d" % chunk_size)
    if not path.is_file():
        raise HashingError("文件不存在：%s" % path)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
