"""统一中间表示。

HTML 与 PDF 解析后都产出同一套结构，下游的切分、索引、检索
完全不感知原始格式（见 docs/spec/02-architecture.md 第 6.1 节）。

四个结构的分工：

    Block            解析出的一个内容块
    ParsedDocument   一份文档的完整解析结果，按文档存进 MongoDB
    DocumentMeta     文档级业务元数据，对应 MySQL 的 documents 表
    Chunk            父块或子块，对应 MySQL 的 chunks 表

所有结构都不可变（frozen）。这些对象会在链路里被多个模块读取，
允许就地修改会让「谁改的、什么时候改的」变得难以追踪。
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BlockType(str, Enum):
    """内容块的类型。"""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"


class SourceFormat(str, Enum):
    """原始文档的格式。"""

    HTML = "html"
    PDF = "pdf"


class ChunkLevel(str, Enum):
    """分块的层级。"""

    PARENT = "parent"
    CHILD = "child"


def new_doc_id() -> str:
    """生成文档标识。

    在文档进入流水线时调用一次，之后的 MongoDB、MySQL、Milvus
    都引用同一个值。这样即使 MongoDB 写入早于 MySQL，
    两边也能关联上。
    """
    return str(uuid4())


def make_parent_chunk_id(doc_id: str, index: int) -> str:
    """构造父块标识，形如 {doc_id}_p0003。"""
    return "%s_p%04d" % (doc_id, index)


def make_child_chunk_id(doc_id: str, index: int) -> str:
    """构造子块标识，形如 {doc_id}_c0012。

    序号左侧补零到四位，让字符串排序与数值排序一致，
    调试时按序查看不会出现 10 排在 2 前面的情况。
    """
    return "%s_c%04d" % (doc_id, index)


class Block(BaseModel):
    """解析出的一个内容块。

    保留原始结构（标题层级、顺序、PDF 页码）是为了在切分时
    能按语义边界而不是纯字符数来断句。
    """

    model_config = ConfigDict(frozen=True)

    type: BlockType = Field(description="块类型")
    text: str = Field(min_length=1, description="块的正文")
    order: int = Field(ge=0, description="在文档内的顺序，从 0 开始")
    level: int = Field(default=0, ge=0, description="标题层级，0 表示非标题")
    page: int | None = Field(default=None, ge=1, description="PDF 页码，HTML 为 None")


class ParsedDocument(BaseModel):
    """一份文档的完整解析结果。

    只放解析产出与技术信息；来源、国家、生效日期这类业务元数据
    在 DocumentMeta 里，两者通过 doc_id 关联。
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1, description="文档标识，由 new_doc_id 生成")
    title: str = Field(min_length=1, description="文档标题")
    source_format: SourceFormat = Field(description="原始格式")
    source_path: Path = Field(description="原始文件路径，必须是绝对路径")
    parser_version: str = Field(min_length=1, description="解析器版本，便于回溯解析质量")
    parsed_at: datetime = Field(description="解析时间")
    blocks: list[Block] = Field(min_length=1, description="解析出的内容块，至少一个")

    @model_validator(mode="after")
    def _require_absolute_source_path(self) -> "ParsedDocument":
        """原始文件路径必须是绝对路径。

        相对路径会随运行目录变化而失效，在 IDE 里尤其容易踩到，
        因此在这里就拦住（见 CLAUDE.md 5.2）。
        """
        if not Path(self.source_path).is_absolute():
            raise ValueError(
                "source_path 必须是绝对路径，收到：%s" % self.source_path
            )
        return self


class DocumentMeta(BaseModel):
    """文档级业务元数据。对应 MySQL 的 documents 表。

    必填字段之外的项允许为空：手工整理语料时可以先导入、后补齐，
    门禁依据 missing_required_fields() 决定能否进入向量库。
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    collected_date: date = Field(description="采集日期，由程序自动写入")
    source_url: str | None = Field(default=None, description="原文出处，缺失则无法进入向量库")
    publisher: str | None = Field(default=None, description="发布机构代码")
    country: str | None = Field(default=None, description="国家代码，EU 表示欧盟整体")
    doc_type: str | None = Field(default=None, description="文档类型代码")
    effective_date: date | None = Field(
        default=None,
        description="生效日期。允许为空，但引用到这类文档时回答必须标注",
    )
    platform: str = Field(default="amazon", min_length=1)

    # 进入向量库前必须齐备的字段。
    # effective_date 不在其中：很多欧盟指南不标注生效日期，
    # 强制要求会让大量文档进不了库。
    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "source_url",
        "publisher",
        "country",
        "doc_type",
    )

    def missing_required_fields(self) -> list[str]:
        """返回仍然缺失的必填字段名。

        门禁依据这个清单决定文档能否进入向量库，以及界面上
        「还差哪些文档没补齐」的提示内容。
        """
        return [
            name for name in self.REQUIRED_FIELDS if getattr(self, name) is None
        ]


class Chunk(BaseModel):
    """一个父块或子块。对应 MySQL 的 chunks 表。

    不带业务元数据（国家、文档类型等）——那些属于文档级，
    由 indexing 阶段把 DocumentMeta 与 Chunk 组合成 Milvus 记录，
    避免同一份元数据在几十个 Chunk 对象里重复。
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    level: ChunkLevel
    chunk_index: int = Field(ge=0, description="同级内的序号，从 0 开始")
    text: str = Field(min_length=1)
    token_count: int = Field(gt=0, description="token 数，便于排查切分异常")
    parent_id: str | None = Field(default=None, description="父块标识，父块自身为空")
    start_offset: int | None = Field(
        default=None, ge=0, description="在父块正文中的起始字符位置，子块必填"
    )
    end_offset: int | None = Field(
        default=None, ge=0, description="在父块正文中的结束字符位置，左闭右开，子块必填"
    )

    @model_validator(mode="after")
    def _validate_level_structure(self) -> "Chunk":
        """按层级校验结构。

        父块没有上位块，因此不该有 parent_id 与偏移量；
        子块必须有 parent_id 与偏移量，偏移量用于在父块正文里
        高亮命中的位置，缺了界面就定位不了
        （见 docs/spec/02-architecture.md 第 6.5 节）。
        """
        if self.level is ChunkLevel.PARENT:
            if self.parent_id is not None:
                raise ValueError("父块不应有 parent_id，收到：%s" % self.parent_id)
            if self.start_offset is not None or self.end_offset is not None:
                raise ValueError("父块不应有字符偏移")
            return self

        if self.parent_id is None:
            raise ValueError("子块必须指定 parent_id，否则检索后无从折叠回完整上下文")
        if self.start_offset is None or self.end_offset is None:
            raise ValueError("子块必须记录相对父块的字符偏移")
        if self.end_offset <= self.start_offset:
            raise ValueError(
                "end_offset（%d）必须大于 start_offset（%d）"
                % (self.end_offset, self.start_offset)
            )
        return self
