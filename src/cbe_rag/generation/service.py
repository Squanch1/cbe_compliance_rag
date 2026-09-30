"""问答编排。

检索、判质量、生成——或者在质量不够时就地拒答。拒答是一条与正常生成
平级的路径（见 docs/spec/02-architecture.md 6.3），不是提示词里的一句
嘱咐。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.generation.answer import Answer, generate_answer
from cbe_rag.generation.citations import CitationReport
from cbe_rag.retrieval.models import RetrievalQuery, RetrievalResult
from cbe_rag.retrieval.service import passes_prefilter, retrieve
from cbe_rag.storage.bailian_client import BailianClient
from cbe_rag.storage.embedding import EmbeddingStore
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.mysql_store import MysqlStore

# 因检索质量不足而拒答时给出的文本。与「模型没写出引用」的降级文案
# 同一句：两者对使用者是同一件事——这段材料答不了这个问题。
REFUSAL_TEXT = "依据不足，无法回答。检索到的材料里没有能支撑这个问题的内容。"


@dataclass(frozen=True)
class QaResult:
    """一次问答的完整结果。

    answer.text 是最终要展示的文本，无论走的哪条路径。

    refused 与 answer.degraded 是两件事，不要混：
    - refused：检索质量不够，**根本没调模型**
    - degraded：调了模型，但它没有给出可核对出处的结论

    分开记是因为排查方向不同：前者要调检索，后者要调提示词或模型。
    """

    answer: Answer
    retrieval: RetrievalResult
    refused: bool
    notes: list[str] = field(default_factory=list)


def answer_question(
    query: RetrievalQuery,
    *,
    config: RetrievalConfig,
    mysql: MysqlStore,
    milvus: MilvusStore,
    embedding: EmbeddingStore,
    client: BailianClient,
) -> QaResult:
    """检索、判质量、生成——或者直接拒答。

    只收 `RetrievalConfig` 而不是整个 `Settings`：这一步用不到别的段，
    收窄参数能让测试直接构造配置，不必先把密码之类的项凑齐。

    **质量不够时根本不调模型。** 调了再让它「别说」是另一回事：那要先
    把材料塞进上下文、付一次调用的钱，然后指望模型听话。真正该做的是一
    开始就不问。
    """
    result = retrieve(
        query, milvus=milvus, mysql=mysql, embedding=embedding, config=config
    )

    if not passes_prefilter(result, config):
        top = result.top_score
        detail = "检索分数低于粗筛线（最高余弦 %s，线 %.4f），未调用模型。" % (
            "无命中" if top is None else "%.4f" % top,
            config.refuse_threshold,
        )
        return QaResult(
            answer=Answer(
                text=REFUSAL_TEXT,
                raw_text="",
                citations=CitationReport(cited=[], invalid_numbers=[]),
                degraded=True,
                notes=[detail],
            ),
            retrieval=result,
            refused=True,
            notes=[detail],
        )

    answer = generate_answer(query.text, result.parents, client)
    return QaResult(answer=answer, retrieval=result, refused=False)
