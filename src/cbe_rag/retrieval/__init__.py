"""检索：把问题变成一组可以送进提示词的父块。

混合检索两路召回（稠密 + 稀疏），加权求和融合，按 parent_id 折叠，
再取回父块全文与引用元数据。
"""

from cbe_rag.retrieval.context import ContextError, load_parents
from cbe_rag.retrieval.models import (
    ParentHit,
    RetrievalQuery,
    RetrievalResult,
    RetrievedParent,
    SearchOutcome,
)
from cbe_rag.retrieval.search import build_filter, fold_by_parent, search
from cbe_rag.retrieval.service import (
    UncalibratedThresholdError,
    is_evidence_sufficient,
    retrieve,
)

__all__ = [
    "ContextError",
    "ParentHit",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedParent",
    "SearchOutcome",
    "UncalibratedThresholdError",
    "build_filter",
    "fold_by_parent",
    "is_evidence_sufficient",
    "load_parents",
    "retrieve",
    "search",
]
