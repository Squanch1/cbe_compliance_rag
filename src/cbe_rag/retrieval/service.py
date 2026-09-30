"""检索编排与粗筛。

一次检索分三步：混合检索与折叠、按 parent_id 恢复父块、判断值不值得交给模型。
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
    """粗筛线还没标定，没法判断哪些请求该拦。"""


def retrieve(
    query: RetrievalQuery,
    *,
    milvus: MilvusStore,
    mysql: MysqlStore,
    embedding: EmbeddingStore,
    config: RetrievalConfig,
) -> RetrievalResult:
    """检索并组装上下文，返回结果与分数。

    **不在这里判拒答**，只把分数带出去。拒答是一条与正常生成平级的路径
    （见 02-architecture 6.3），它属于上层的编排；塞进检索层会让「检索」
    这一个动作凭空多出一种结局。
    """
    outcome = search(query, milvus=milvus, embedding=embedding, config=config)
    return RetrievalResult(
        parents=load_parents(outcome.parents, mysql),
        top_score=outcome.dense_top_score,
    )


def passes_prefilter(result: RetrievalResult, config: RetrievalConfig) -> bool:
    """粗筛：这份检索结果值不值得交给模型。

    **它不判断「材料够不够回答」**——那是模型的事。这里只拦「明显无关」的
    问题，省下一次调用。

    为什么不让它做判断题：稠密相似度**分不开「主题相近但没答案」和
    「有答案」**。实测里问「欧盟进口货物的关税起征点」（语料里只有增值税，
    没有关税）拿到 0.6451，而问「IOSS 适用金额上限」（语料里明确有 150
    欧元）拿到 0.5867——前者反而更高。这是稠密向量的固有限制：它看的是
    语义像不像，不包含「有没有答案」这个信息。

    所以这条线定得**保守**：宁可漏拦，不可误伤。漏拦只是多调一次模型，
    误伤是用户拿不到本来能答的问题。真正判断「材料够不够」由模型做——
    实测它在这方面比稠密向量靠谱（问德国税率时它自己说了「未提及」）。

    **阈值未标定时报错，而不是放行。** 未经校准的线拦不住该拦的，
    配置里对这一点有明确约定（见 docs/spec/02-architecture.md 6.3）。
    """
    threshold = config.refuse_threshold
    if threshold is None:
        raise UncalibratedThresholdError(
            "粗筛线尚未标定（CBE_RETRIEVAL__REFUSE_THRESHOLD 为空），"
            "无法判断哪些请求该拦。先在评测集上标定再启用。"
        )

    if result.top_score is None:
        # 一条都没检索到，那是真的无关
        return False
    return result.top_score >= threshold
