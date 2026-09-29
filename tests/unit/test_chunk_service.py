"""父块与子块组装的单元测试。

前面两步各自验证了切分逻辑，这一步验证的是**把结果拼成 Chunk 对象**
——父子关系、编号、偏移——是否正确。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from cbe_rag.ingestion.chunker.counter import TokenCounter
from cbe_rag.ingestion.chunker.service import chunk_document
from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ChunkLevel,
    ParsedDocument,
    SourceFormat,
)

DOC_ID = "6f1c2f7e-1a2b-4c3d-8e9f-0a1b2c3d4e5f"


class FakeTokenizer:
    """每个字符算一个 token。"""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(range(len(text)))


def make_document(*specs: tuple[BlockType, str, int]) -> ParsedDocument:
    """specs 元素为 (块类型, 文本, 标题层级)。"""
    blocks = [
        Block(type=kind, text=text, order=index, level=level)
        for index, (kind, text, level) in enumerate(specs)
    ]
    return ParsedDocument(
        doc_id=DOC_ID,
        title="测试文档",
        source_format=SourceFormat.HTML,
        source_path=Path("C:/proj/data/raw/x.html"),
        parser_version="0.1.0",
        parsed_at=datetime(2026, 9, 28),
        blocks=blocks,
    )


def build_counter() -> TokenCounter:
    return TokenCounter(Path("unused"), tokenizer=FakeTokenizer())


def paragraph(text: str) -> tuple[BlockType, str, int]:
    return (BlockType.PARAGRAPH, text, 0)


def heading(text: str) -> tuple[BlockType, str, int]:
    return (BlockType.HEADING, text, 1)


CONST = dict(parent_lower=30, parent_upper=60, child_target=10)


class TestAssembly:
    def test_produces_both_levels(self) -> None:
        document = make_document(heading("标题"), paragraph("内容内容内容内容"))

        chunks = chunk_document(document, build_counter(), **CONST)

        levels = {c.level for c in chunks}
        assert levels == {ChunkLevel.PARENT, ChunkLevel.CHILD}

    def test_children_point_at_their_parent(self) -> None:
        document = make_document(paragraph("甲" * 50))

        chunks = chunk_document(document, build_counter(), **CONST)

        parents = [c for c in chunks if c.level is ChunkLevel.PARENT]
        children = [c for c in chunks if c.level is ChunkLevel.CHILD]
        assert parents and children
        for child in children:
            assert child.parent_id == parents[0].chunk_id

    def test_parent_has_no_parent_id(self) -> None:
        document = make_document(paragraph("甲" * 50))

        parents = [
            c
            for c in chunk_document(document, build_counter(), **CONST)
            if c.level is ChunkLevel.PARENT
        ]

        assert all(p.parent_id is None for p in parents)

    def test_doc_id_is_carried_to_every_chunk(self) -> None:
        document = make_document(paragraph("甲" * 50))

        chunks = chunk_document(document, build_counter(), **CONST)

        assert all(c.doc_id == DOC_ID for c in chunks)


class TestIndexing:
    def test_parent_indices_are_contiguous(self) -> None:
        document = make_document(
            heading("标题一"), paragraph("甲" * 40), heading("标题二"), paragraph("乙" * 40)
        )

        chunks = chunk_document(document, build_counter(), **CONST)

        indices = [c.chunk_index for c in chunks if c.level is ChunkLevel.PARENT]
        assert indices == list(range(len(indices)))

    def test_child_indices_are_contiguous(self) -> None:
        document = make_document(paragraph("甲" * 100))

        chunks = chunk_document(document, build_counter(), **CONST)

        indices = [c.chunk_index for c in chunks if c.level is ChunkLevel.CHILD]
        assert indices == list(range(len(indices)))

    def test_chunk_ids_follow_the_convention(self) -> None:
        document = make_document(paragraph("甲" * 50))

        chunks = chunk_document(document, build_counter(), **CONST)

        for chunk in chunks:
            if chunk.level is ChunkLevel.PARENT:
                assert chunk.chunk_id == "%s_p%04d" % (DOC_ID, chunk.chunk_index)
            else:
                assert chunk.chunk_id == "%s_c%04d" % (DOC_ID, chunk.chunk_index)


class TestOffsets:
    def test_child_text_is_a_slice_of_its_parent(self) -> None:
        # 这条不变式是界面高亮的前提，必须成立
        document = make_document(paragraph("甲" * 45), paragraph("乙" * 45))

        chunks = chunk_document(document, build_counter(), **CONST)

        by_id = {c.chunk_id: c for c in chunks}
        for child in chunks:
            if child.level is not ChunkLevel.CHILD:
                continue
            parent = by_id[child.parent_id]
            assert parent.text[child.start_offset : child.end_offset] == child.text

    def test_parent_has_no_offsets(self) -> None:
        document = make_document(paragraph("甲" * 50))

        parents = [
            c
            for c in chunk_document(document, build_counter(), **CONST)
            if c.level is ChunkLevel.PARENT
        ]

        assert all(p.start_offset is None and p.end_offset is None for p in parents)


class TestTokenCounts:
    def test_every_chunk_records_a_positive_count(self) -> None:
        document = make_document(paragraph("甲" * 50))

        chunks = chunk_document(document, build_counter(), **CONST)

        assert all(c.token_count > 0 for c in chunks)

    def test_parent_count_covers_its_children(self) -> None:
        # 父块的 token 数应大于其中任一子块——否则父子关系就没有意义
        document = make_document(paragraph("甲" * 45), paragraph("乙" * 45))

        chunks = chunk_document(document, build_counter(), **CONST)

        by_id = {c.chunk_id: c for c in chunks}
        for child in chunks:
            if child.level is not ChunkLevel.CHILD:
                continue
            assert by_id[child.parent_id].token_count >= child.token_count


class TestEdgeCases:
    def test_single_block_document(self) -> None:
        document = make_document(paragraph("只有一段内容"))

        chunks = chunk_document(document, build_counter(), **CONST)

        assert len([c for c in chunks if c.level is ChunkLevel.PARENT]) == 1
        assert len([c for c in chunks if c.level is ChunkLevel.CHILD]) == 1

    def test_no_content_is_lost(self) -> None:
        # 父子两级加起来不该丢文字：把所有子块拼起来应覆盖父块的每一段
        document = make_document(paragraph("甲" * 30), paragraph("乙" * 30))

        chunks = chunk_document(document, build_counter(), **CONST)

        parents = [c for c in chunks if c.level is ChunkLevel.PARENT]
        children = [c for c in chunks if c.level is ChunkLevel.CHILD]
        assert len(children) >= len(parents)
        for child in children:
            assert len(child.text) > 0
