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
class ChunkHit:
    """Milvus 返回的一条子块命中。

    score 是融合后的分数，不是任何一路的原始分——原始分量纲不同，
    没法放在一起比较。
    """

    chunk_id: str
    doc_id: str
    score: float


@dataclass(frozen=True)
class SearchOutcome:
    """一次混合检索的原始产出。

    dense_top_score 单独取出来，是因为拒答判据用的是它：稠密路的最高
    余弦相似度有绝对含义（0.6 就是 0.6），而融合分经过归一化和加权，
    数值随权重变化，拿它当阈值没有可比性。
    """

    hits: list[ChunkHit]
    dense_top_score: float | None


@dataclass(frozen=True)
class ParentHit:
    """折叠后的父块命中。

    score 取该父块下所有命中子块的最高分。一个父块的多个子块都命中，
    说明它整体相关，最高分只是它的下界；求平均会被不相关的子块拉低。

    matched_children 是命中的子块数，只用于排查，不参与排序。
    """

    parent_id: str
    doc_id: str
    score: float
    matched_children: int = 1


@dataclass(frozen=True)
class RetrievedParent:
    """一条可以直接送进提示词的父块，连同引用所需的元数据。

    元数据从 documents 表按 doc_id 取回。缺 source_url 的文档进不了
    向量库，因此这里理论上不会为空——但类型上仍允许，生成层要能处理。
    """

    parent_id: str
    doc_id: str
    score: float
    text: str
    title: str
    source_url: str | None
    country: str | None
    doc_type: str | None
    publisher: str | None
    effective_date: date | None


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的最终结果。

    top_score 是稠密路最高余弦相似度，交给拒答判定；没有任何命中时为 None。
    """

    parents: list[RetrievedParent] = field(default_factory=list)
    top_score: float | None = None
