"""检索层的数据结构。

按流程顺序排：

    RetrievalQuery   输入
    SearchOutcome    两路检索的原始产出
    ChunkHit         子块命中
    ParentHit        折叠后的父块命中
    RetrievedParent  可以直接送进提示词的父块
    RetrievalResult  最终结果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class RetrievalQuery:
    """一次检索的输入。

    三个过滤维度由**调用方显式传**，不从问题里猜：猜错了会直接筛掉正确
    文档，而且不报错。为 None 表示不按该维度筛。
    """

    text: str
    country: str | None = None
    doc_type: str | None = None
    publisher: str | None = None


@dataclass(frozen=True)
class SearchOutcome:
    """一次混合检索的产出：折叠后的父块命中，加一个质量指标。

    parents 已按分数从高到低排好，且不超过配置的 context_parents 条。
    折叠在这里做完，不留给调用方：子块是检索单位，父块才是送进提示词的
    单位，中间这一步不落地，就没法解释「为什么召回的是这几个」。

    dense_top_score 单独取出来，是因为拒答判据用的是它：稠密路的最高
    余弦相似度有绝对含义（0.6 就是 0.6），而融合分经过归一化和加权，
    数值随权重变化，拿它当阈值没有可比性。
    """

    parents: list[ParentHit]
    dense_top_score: float | None


@dataclass(frozen=True)
class MatchedChild:
    """一条命中的子块。

    引用最终锚定到它，而不是父块：模型读的是父块全文，但真正支撑某句
    结论的可能只是其中一段。只锚定分数最高的那条，用户点开会发现那段
    原文支撑不了那句话——可追溯性静默失效，而且不报错。

    原文与字符偏移不在这里：Milvus 里没存这两样，要按 chunk_id 回 MySQL
    取，见 retrieval/context.py。
    """

    chunk_id: str
    score: float


@dataclass(frozen=True)
class ParentHit:
    """折叠后的父块命中。

    score 取该父块下所有命中子块的最高分。一个父块的多个子块都命中，
    说明它整体相关，最高分只是它的下界；求平均会被不相关的子块拉低。

    **children 保留全部命中的子块，不只是最高分那条。** 排序用最高分，
    但引用要高亮的是「哪几段被命中了」，只留一条会漏掉真正支撑结论的
    那一段（见 docs/spec/02-architecture.md 6.5）。
    """

    parent_id: str
    doc_id: str
    score: float
    children: list[MatchedChild] = field(default_factory=list)

    @property
    def matched_children(self) -> int:
        """命中的子块数。"""
        return len(self.children)


@dataclass(frozen=True)
class RetrievedChild:
    """一条命中的子块，连同可以高亮的定位信息。

    偏移指向**父块正文里的位置**，不是全文里的位置——界面拿到的就是
    父块正文，按这个偏移高亮即可。

    Milvus 里没有原文和偏移，这两样是回 MySQL 取的。
    """

    chunk_id: str
    score: float
    text: str
    start_offset: int | None
    end_offset: int | None


@dataclass(frozen=True)
class RetrievedParent:
    """一条可以直接送进提示词的父块，连同引用所需的元数据。

    元数据从 documents 表按 doc_id 取回。缺 source_url 的文档进不了
    向量库，因此这里理论上不会为空——但类型上仍允许，生成层要能处理。

    matched_children 是这条父块下全部命中的子块，不只是分数最高的那条：
    界面要靠它们高亮，而模型可能依据其中任意一条作答。
    """

    parent_id: str
    doc_id: str
    score: float
    text: str
    token_count: int
    title: str
    source_url: str | None
    country: str | None
    doc_type: str | None
    publisher: str | None
    effective_date: date | None
    matched_children: list[RetrievedChild] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的最终结果。

    top_score 是稠密路最高余弦相似度，交给拒答判定；没有任何命中时为 None。
    """

    parents: list[RetrievedParent] = field(default_factory=list)
    top_score: float | None = None
