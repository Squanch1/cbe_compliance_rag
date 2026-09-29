"""混合检索与折叠。

把问题编码成两路向量，交给 Milvus 做融合检索，再按 parent_id 折叠成父块。
子块是检索单位，父块才是送进提示词的单位，这一步就是把前者换成后者。
"""

from __future__ import annotations

import re

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.models import ParentHit, RetrievalQuery, SearchOutcome
from cbe_rag.storage.embedding import EmbeddingStore
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.records import VectorHit

# 三个过滤维度的取值形状：国家代码、文档类型代码、发布机构代码都是
# 字母数字加下划线连字符（EU、guideline、eu_commission）。
#
# Milvus 的 filter 没有参数占位符，值只能拼进字符串，而它最终来自接口层，
# 因此拼之前先按形状校验。
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_-]+$")


def build_filter(query: RetrievalQuery) -> str:
    """把过滤条件拼成 Milvus 的表达式，没有条件时返回空串。

    只拼调用方显式传的维度。从问题里猜国家会直接筛掉正确文档，而且错得
    不报错——省下的那点调用方负担，不值这个风险。
    """
    clauses: list[str] = []
    for field, value in (
        ("country", query.country),
        ("doc_type", query.doc_type),
        ("publisher", query.publisher),
    ):
        if value is None:
            continue
        if not _SAFE_VALUE.match(value):
            raise ValueError(
                "%s 的取值含非法字符，无法安全拼进 filter：%r" % (field, value)
            )
        clauses.append('%s == "%s"' % (field, value))
    return " and ".join(clauses)


def fold_by_parent(hits: list[VectorHit]) -> list[ParentHit]:
    """按 parent_id 折叠，同一父块只保留最高分，返回按分数降序的结果。

    **取最高分而不是求平均。** 一个父块的多个子块都命中，说明它整体
    相关，最高分只是它的下界；求平均会被其中不相关的子块拉低，反而
    惩罚了覆盖得广的父块。

    分数相同时按 parent_id 排，保证同一份语料两次跑出的顺序一致——
    顺序随插入顺序变会让问题难以复现。
    """
    best: dict[str, ParentHit] = {}
    for hit in hits:
        current = best.get(hit.parent_id)
        if current is None:
            best[hit.parent_id] = ParentHit(
                parent_id=hit.parent_id,
                doc_id=hit.doc_id,
                score=hit.score,
                matched_children=1,
            )
            continue
        best[hit.parent_id] = ParentHit(
            parent_id=current.parent_id,
            doc_id=current.doc_id,
            score=max(current.score, hit.score),
            matched_children=current.matched_children + 1,
        )

    return sorted(best.values(), key=lambda item: (-item.score, item.parent_id))


def search(
    query: RetrievalQuery,
    *,
    milvus: MilvusStore,
    embedding: EmbeddingStore,
    config: RetrievalConfig,
) -> SearchOutcome:
    """检索、融合、折叠，返回不超过 context_parents 条父块命中。

    融合后取回两路候选数之和，而不是直接就取 context_parents：折叠会
    让条数变少（多个子块属于同一个父块），先砍一批再折叠，剩下的父块
    可能不够用。
    """
    encoded = embedding.encode([query.text])
    dense = encoded.dense[0]
    sparse = encoded.sparse[0]
    filter_expression = build_filter(query)

    hits = milvus.hybrid_search(
        dense,
        sparse,
        dense_limit=config.dense_limit,
        sparse_limit=config.sparse_limit,
        weights=(config.dense_weight, config.sparse_weight),
        limit=config.dense_limit + config.sparse_limit,
        filter_expression=filter_expression,
    )

    return SearchOutcome(
        parents=fold_by_parent(hits)[: config.context_parents],
        dense_top_score=milvus.top_dense_score(
            dense, filter_expression=filter_expression
        ),
    )
