"""索引：把解析产物向量化并写入检索库。

负责判重、解析、切块、向量化、落库的编排。批量入口是
service.import_documents。
"""

from cbe_rag.indexing.decision import decide
from cbe_rag.indexing.hashing import HashingError, file_content_hash
from cbe_rag.indexing.models import (
    Decision,
    DocumentOutcome,
    ImportAction,
    ImportReport,
    count_by_action,
)

__all__ = [
    "Decision",
    "DocumentOutcome",
    "HashingError",
    "ImportAction",
    "ImportReport",
    "count_by_action",
    "decide",
    "file_content_hash",
]
