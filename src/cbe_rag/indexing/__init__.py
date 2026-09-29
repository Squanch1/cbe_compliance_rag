"""索引：把解析产物向量化并写入检索库。

负责判重、解析、切块、向量化、落库的编排。批量入口是
service.import_documents。
"""

from cbe_rag.indexing.hashing import HashingError, file_content_hash

__all__ = [
    "HashingError",
    "file_content_hash",
]
