"""从存储读出、或将要写入存储的业务记录。

与 ddl.py 的分工：ddl.py 回答「表长什么样」，本模块回答「一行是什么」。

定义在 storage 层而不是 indexing 层，是因为它由适配器返回、也由适配器
接收——适配器不该反过来依赖调用方。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from cbe_rag.storage.ddl import DocumentStatus


@dataclass(frozen=True)
class DocumentRecord:
    """`documents` 表里的一行。

    读与写共用同一个类型：两者字段完全一致，分成两个再互相转换，
    除了多几处赋值没有别的作用。

    **不直接嵌入 `DocumentMeta`**：那样 storage 层就要 import 解析层的
    类型，而依赖方向是解析 -> 索引 -> 存储。字段平铺虽然多写几行，但
    保住了方向，也让「表列与字段一一对应」这件事一眼可见。
    """

    doc_id: str
    content_hash: str
    status: DocumentStatus
    title: str
    platform: str
    source_url: str | None
    publisher: str | None
    country: str | None
    doc_type: str | None
    effective_date: date | None
    collected_date: date
    raw_path: str
    # 元数据齐备时为空元组，这也是最常见的情况
    missing_fields: tuple[str, ...] = ()
    # 各解析层的尝试记录，落库时转成 JSON。
    #
    # 只有真正进过解析的文档才有内容——元数据不齐而没切分的那些这里是空的。
    # 每条记「用哪个工具、为什么没过」，外加质量评估的原始分数：调阈值时
    # 要看分布，而分布只能从这些分数里来。
    parse_attempts: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Dimension:
    """维度表里的一行。

    三张维度表的字段一致（代码、中英文名），因此共用一个类型；发布机构
    那张多一个 official_url，界面用不上，不在这里出现。
    """

    code: str
    name_zh: str
    name_en: str


@dataclass(frozen=True)
class Dimensions:
    """三张维度表的可用取值，供界面渲染选项。

    **代码与中文名都要带上**：界面显示中文给使用者看，提交的却是代码——
    代码才是后端拼进检索表达式的东西。只给中文名的话，提交过去就是错的；
    只给代码，界面上就是现在这样一堆英文。
    """

    countries: list[Dimension] = field(default_factory=list)
    doc_types: list[Dimension] = field(default_factory=list)
    publishers: list[Dimension] = field(default_factory=list)


@dataclass(frozen=True)
class VectorHit:
    """向量检索的一条命中。

    **带 parent_id 是必需的，不能从 chunk_id 推。** 子块的编号是全局
    递增的（`{doc_id}_c0042`），与它所属父块的编号（`{doc_id}_p0007`）
    没有对应关系，只能由 Milvus 一并返回。

    score 是两路融合后的分数。各路的原始分不在这里——量纲不同（同一个
    问题下稠密 0.5867、稀疏 0.0786），放在一起很容易被拿去比较或当阈值。
    需要用原始分的地方（拒答判据）单独取，见 MilvusStore.top_dense_score。
    """

    chunk_id: str
    doc_id: str
    parent_id: str
    score: float


@dataclass(frozen=True)
class ChunkVector:
    """一条待写入 Milvus 的子块记录。

    **只有子块**：父块不建向量，检索召回子块后靠 parent_id 回 MySQL
    取全文送进提示词（small-to-big）。父块存进 Milvus 只是白占空间。

    country / doc_type / publisher 在这里冗余存一份，是为了让检索能
    直接在 Milvus 里按维度过滤，不必先回 MySQL 查一遍。
    """

    chunk_id: str
    doc_id: str
    parent_id: str
    chunk_index: int
    country: str
    doc_type: str
    publisher: str
    dense: list[float]
    sparse: dict[int, float]
