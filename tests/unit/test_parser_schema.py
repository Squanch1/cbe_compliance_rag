"""统一中间表示的数据结构测试。

这些结构是整个下游链路的基础，字段与约束一旦定错，切分、索引、
检索都要返工，因此约束部分测得细一些。
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    Chunk,
    ChunkLevel,
    DocumentMeta,
    ParsedDocument,
    SourceFormat,
    make_child_chunk_id,
    make_parent_chunk_id,
    new_doc_id,
)

DOC_ID = "6f1c2f7e-1a2b-4c3d-8e9f-0a1b2c3d4e5f"


def make_block(**overrides: object) -> Block:
    payload: dict[str, object] = {
        "type": BlockType.PARAGRAPH,
        "text": "Import One-Stop Shop applies to consignments not exceeding 150 EUR.",
        "order": 0,
    }
    payload.update(overrides)
    return Block(**payload)  # type: ignore[arg-type]


def make_parent(**overrides: object) -> Chunk:
    payload: dict[str, object] = {
        "chunk_id": make_parent_chunk_id(DOC_ID, 0),
        "doc_id": DOC_ID,
        "level": ChunkLevel.PARENT,
        "chunk_index": 0,
        "text": "父块正文",
        "token_count": 1500,
    }
    payload.update(overrides)
    return Chunk(**payload)  # type: ignore[arg-type]


def make_child(**overrides: object) -> Chunk:
    payload: dict[str, object] = {
        "chunk_id": make_child_chunk_id(DOC_ID, 0),
        "doc_id": DOC_ID,
        "parent_id": make_parent_chunk_id(DOC_ID, 0),
        "level": ChunkLevel.CHILD,
        "chunk_index": 0,
        "text": "子块正文",
        "token_count": 300,
        "start_offset": 0,
        "end_offset": 4,
    }
    payload.update(overrides)
    return Chunk(**payload)  # type: ignore[arg-type]


class TestBlock:
    def test_minimal_block(self) -> None:
        block = make_block()

        assert block.type is BlockType.PARAGRAPH
        assert block.level == 0
        assert block.page is None

    def test_heading_carries_level(self) -> None:
        block = make_block(type=BlockType.HEADING, level=2)

        assert block.level == 2

    def test_empty_text_is_rejected(self) -> None:
        # 空块对下游没有意义，应在构造时就失败而不是留到切分阶段
        with pytest.raises(ValidationError):
            make_block(text="")

    def test_negative_order_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_block(order=-1)

    def test_pdf_page_is_optional_but_positive(self) -> None:
        assert make_block(page=3).page == 3
        with pytest.raises(ValidationError):
            make_block(page=0)

    def test_is_immutable(self) -> None:
        block = make_block()

        with pytest.raises(ValidationError):
            block.text = "改一下"  # type: ignore[misc]


class TestParsedDocument:
    def test_minimal_document(self) -> None:
        document = ParsedDocument(
            doc_id=DOC_ID,
            title="IOSS 指南",
            source_format=SourceFormat.PDF,
            source_path="C:/proj/data/raw/ioss.pdf",
            parser_version="0.1.0",
            parsed_at=datetime(2026, 9, 28, 10, 0, 0),
            blocks=[make_block()],
        )

        assert document.doc_id == DOC_ID
        assert document.source_format is SourceFormat.PDF
        assert len(document.blocks) == 1

    def test_empty_block_list_is_rejected(self) -> None:
        # 解析出零个块，说明解析失败而不是文档为空
        with pytest.raises(ValidationError):
            ParsedDocument(
                doc_id=DOC_ID,
                title="空文档",
                source_format=SourceFormat.HTML,
                source_path="C:/proj/data/raw/x.html",
                parser_version="0.1.0",
                parsed_at=datetime(2026, 9, 28),
                blocks=[],
            )

    def test_relative_source_path_is_rejected(self) -> None:
        # 相对路径会随运行目录变化而失效，在 IDE 里尤其容易踩到
        # （见 CLAUDE.md 5.2 的路径约定）
        with pytest.raises(ValidationError):
            ParsedDocument(
                doc_id=DOC_ID,
                title="x",
                source_format=SourceFormat.HTML,
                source_path="data/raw/x.html",
                parser_version="0.1.0",
                parsed_at=datetime(2026, 9, 28),
                blocks=[make_block()],
            )

    def test_source_format_is_restricted(self) -> None:
        with pytest.raises(ValidationError):
            ParsedDocument(
                doc_id=DOC_ID,
                title="x",
                source_format="docx",  # type: ignore[arg-type]
                source_path="C:/x",
                parser_version="0.1.0",
                parsed_at=datetime(2026, 9, 28),
                blocks=[make_block()],
            )


class TestDocumentMeta:
    def make_meta(self, **overrides: object) -> DocumentMeta:
        payload: dict[str, object] = {
            "doc_id": DOC_ID,
            "title": "IOSS 指南",
            "collected_date": date(2026, 9, 28),
        }
        payload.update(overrides)
        return DocumentMeta(**payload)  # type: ignore[arg-type]

    def test_only_collected_date_is_required_beyond_identity(self) -> None:
        meta = self.make_meta()

        assert meta.collected_date == date(2026, 9, 28)
        assert meta.source_url is None
        assert meta.country is None

    def test_effective_date_may_be_absent(self) -> None:
        # 很多欧盟指南不标注生效日期，规范允许为空但回答时要标注
        assert self.make_meta().effective_date is None

    def test_missing_required_fields_are_listed(self) -> None:
        # 门禁依据这个清单决定能否进入向量库
        meta = self.make_meta()

        missing = meta.missing_required_fields()

        assert set(missing) == {"source_url", "publisher", "country", "doc_type"}

    def test_complete_meta_has_no_missing_fields(self) -> None:
        meta = self.make_meta(
            source_url="https://europa.eu/ioss",
            publisher="eu_commission",
            country="EU",
            doc_type="guideline",
        )

        assert meta.missing_required_fields() == []

    def test_effective_date_absence_is_not_reported_as_missing(self) -> None:
        # 生效日期为空不算缺项，否则多数欧盟文档都进不了向量库
        meta = self.make_meta(
            source_url="https://europa.eu/ioss",
            publisher="eu_commission",
            country="EU",
            doc_type="guideline",
        )

        assert "effective_date" not in meta.missing_required_fields()


class TestChunkIdConvention:
    def test_parent_id_format(self) -> None:
        assert make_parent_chunk_id(DOC_ID, 3) == "%s_p0003" % DOC_ID

    def test_child_id_format(self) -> None:
        assert make_child_chunk_id(DOC_ID, 12) == "%s_c0012" % DOC_ID

    def test_padding_keeps_string_order_matching_numeric_order(self) -> None:
        # 补零是为了让字符串排序与数值排序一致，便于调试时按序查看
        ids = [make_child_chunk_id(DOC_ID, i) for i in (2, 10, 1)]

        assert sorted(ids) == [
            make_child_chunk_id(DOC_ID, 1),
            make_child_chunk_id(DOC_ID, 2),
            make_child_chunk_id(DOC_ID, 10),
        ]

    def test_new_doc_id_is_unique(self) -> None:
        assert len({new_doc_id() for _ in range(100)}) == 100


class TestChunkParentStructure:
    def test_parent_has_no_parent_id_and_no_offsets(self) -> None:
        parent = make_parent()

        assert parent.parent_id is None
        assert parent.start_offset is None
        assert parent.end_offset is None

    def test_parent_must_not_declare_parent_id(self) -> None:
        with pytest.raises(ValidationError):
            make_parent(parent_id=make_parent_chunk_id(DOC_ID, 1))

    def test_parent_must_not_declare_offsets(self) -> None:
        # 父块没有上位块，偏移量无从谈起
        with pytest.raises(ValidationError):
            make_parent(start_offset=0, end_offset=10)


class TestChunkChildStructure:
    def test_child_carries_parent_and_offsets(self) -> None:
        child = make_child()

        assert child.parent_id == make_parent_chunk_id(DOC_ID, 0)
        assert (child.start_offset, child.end_offset) == (0, 4)

    def test_child_must_have_parent_id(self) -> None:
        # 没有父块，检索后就无从折叠回完整上下文
        with pytest.raises(ValidationError):
            make_child(parent_id=None)

    def test_child_must_have_offsets(self) -> None:
        # 偏移量用于在父块正文里高亮命中的位置，缺了界面就定位不了
        with pytest.raises(ValidationError):
            make_child(start_offset=None, end_offset=None)

    def test_end_offset_must_exceed_start_offset(self) -> None:
        with pytest.raises(ValidationError):
            make_child(start_offset=10, end_offset=10)

    def test_negative_offset_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_child(start_offset=-1, end_offset=5)


class TestChunkCommons:
    def test_token_count_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            make_child(token_count=0)

    def test_empty_text_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_child(text="")

    def test_is_immutable(self) -> None:
        chunk = make_child()

        with pytest.raises(ValidationError):
            chunk.text = "改一下"  # type: ignore[misc]

    def test_does_not_carry_business_metadata(self) -> None:
        # 国家、文档类型属于文档级，由 indexing 阶段与 Chunk 组合，
        # 不在每个 Chunk 上重复存一份
        fields = set(Chunk.model_fields)

        assert not fields & {"country", "doc_type", "publisher", "source_url"}
