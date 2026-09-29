"""检索：把问题变成一组可以送进提示词的父块。

混合检索两路召回（稠密 + 稀疏），加权求和融合，按 parent_id 折叠，
再取回父块全文与引用元数据。
"""

from cbe_rag.retrieval.models import (
    ChunkHit,
    ParentHit,
    RetrievalQuery,
    RetrievalResult,
    RetrievedParent,
    SearchOutcome,
)

__all__ = [
    "ChunkHit",
    "ParentHit",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedParent",
    "SearchOutcome",
]
