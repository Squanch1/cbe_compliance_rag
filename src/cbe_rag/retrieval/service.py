"""检索编排与拒答判定。

一次检索分三步：混合检索与折叠、按 parent_id 恢复父块、判质量。
"""

from __future__ import annotations

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.context import load_parents
from cbe_rag.retrieval.models import RetrievalQuery, RetrievalResult
from cbe_rag.retrieval.search import search
from cbe_rag.storage.embedding import EmbeddingStore
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.mysql_store import MysqlStore


class UncalibratedThresholdError(Exception):
    """拒答阈值还没标定，无法判断检索质量。"""


def retrieve(
    query: RetrievalQuery,
    *,
    milvus: MilvusStore,
    mysql: MysqlStore,
    embedding: EmbeddingStore,
    config: RetrievalConfig,
) -> RetrievalResult:
    """检索并组装上下文，返回结果与质量判据。

    **不在这里判拒答**，只把判据（top_score）带出去。拒答是一条与正常
    生成平级的路径（见 02-architecture 6.3），它属于上层的编排；塞进
    检索层会让「检索」这一个动作凭空多出一种结局。
    """
    outcome = search(query, milvus=milvus, embedding=embedding, config=config)
    return RetrievalResult(
        parents=load_parents(outcome.parents, mysql),
        top_score=outcome.dense_top_score,
    )


def is_evidence_sufficient(
    result: RetrievalResult, config: RetrievalConfig
) -> bool:
    """判断检索质量够不够生成回答。

    **阈值未标定时报错，而不是放行。** 未经校准的检索结果流到生成层，
    得到的是「看着有出处、其实没检索到相关内容」的答案——比直接拒答
    危险得多，因为使用者无从分辨。配置里对这一点有明确约定（见
    docs/spec/02-architecture.md 6.3）。
    """
    threshold = config.refuse_threshold
    if threshold is None:
        raise UncalibratedThresholdError(
            "拒答阈值尚未标定（CBE_RETRIEVAL__REFUSE_THRESHOLD 为空），"
            "无法判断检索质量。先在评测集上标定再启用。"
        )

    if result.top_score is None:
        # 一条都没检索到，阈值再低也不该放行
        return False
    return result.top_score >= threshold
