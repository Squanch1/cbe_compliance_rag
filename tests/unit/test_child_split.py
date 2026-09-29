"""子块切分的单元测试。

重点验证一条不变式：**每个子块的文本必须等于父块文本在该区间上的切片**。
界面靠这条不变式在父块正文里高亮命中的子块（见 02-architecture 6.5），
一旦偏移算错，高亮就会错位甚至越界。
"""

from __future__ import annotations

import pytest

from cbe_rag.ingestion.chunker.child import (
    BLOCK_SEPARATOR,
    ChildSplitError,
    chunk_parent,
)
from cbe_rag.ingestion.parser.schema import Block, BlockType


def paragraph(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text, order=0)


def blocks_and_counts(specs: list[tuple[Block, int]]) -> tuple[list[Block], list[int]]:
    return [b for b, _ in specs], [c for _, c in specs]


class TestEmptyInput:
    def test_no_blocks_yields_empty_text_and_no_children(self) -> None:
        result = chunk_parent([], [], target=100)

        assert result.text == ""
        assert result.children == []


class TestSpanInvariant:
    """子块必须是父块的连续子串。"""

    def test_each_child_is_a_slice_of_parent_text(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲乙丙"), 50), (paragraph("丁戊己"), 50)])

        result = chunk_parent(blocks, counts, target=50)

        for child in result.children:
            assert result.text[child.start : child.end] == child.text

    def test_spans_are_within_bounds(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph(str(i) * 10), 30) for i in range(6)])

        result = chunk_parent(blocks, counts, target=60)

        for child in result.children:
            assert 0 <= child.start < child.end <= len(result.text)

    def test_spans_are_ordered_and_do_not_overlap(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph(str(i) * 10), 30) for i in range(6)])

        result = chunk_parent(blocks, counts, target=60)

        for earlier, later in zip(result.children, result.children[1:]):
            assert earlier.end <= later.start

    def test_children_cover_the_whole_parent_text(self) -> None:
        # 子块切分不该丢内容
        blocks, counts = blocks_and_counts([(paragraph(str(i) * 10), 30) for i in range(5)])

        result = chunk_parent(blocks, counts, target=60)

        assert result.children[0].start == 0
        assert result.children[-1].end == len(result.text)


class TestGrouping:
    def test_small_blocks_are_merged(self) -> None:
        # 累积不到目标就继续攒
        blocks, counts = blocks_and_counts([(paragraph("甲"), 20), (paragraph("乙"), 20)])

        result = chunk_parent(blocks, counts, target=100)

        assert len(result.children) == 1

    def test_breaks_when_target_reached(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲"), 60), (paragraph("乙"), 60)])

        result = chunk_parent(blocks, counts, target=60)

        assert len(result.children) == 2

    def test_last_partial_group_is_kept(self) -> None:
        # 最后不足目标的余量也要产出，不能丢
        blocks, counts = blocks_and_counts([(paragraph("甲"), 60), (paragraph("乙"), 10)])

        result = chunk_parent(blocks, counts, target=60)

        assert len(result.children) == 2

    def test_single_oversized_block_becomes_its_own_child(self) -> None:
        # 块内没有更小的边界可用，切不开。记录这个限制。
        blocks, counts = blocks_and_counts([(paragraph("巨块"), 500), (paragraph("小"), 10)])

        result = chunk_parent(blocks, counts, target=100)

        assert result.children[0].text == "巨块"


class TestParentText:
    def test_blocks_are_joined_with_the_separator(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲"), 10), (paragraph("乙"), 10)])

        result = chunk_parent(blocks, counts, target=100)

        assert result.text == "甲%s乙" % BLOCK_SEPARATOR

    def test_single_block_text_is_unchanged(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("只有一段"), 10)])

        assert chunk_parent(blocks, counts, target=100).text == "只有一段"

    def test_child_spanning_two_blocks_includes_the_separator(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲"), 10), (paragraph("乙"), 10)])

        result = chunk_parent(blocks, counts, target=100)

        assert result.children[0].text == "甲%s乙" % BLOCK_SEPARATOR


class TestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        blocks, _ = blocks_and_counts([(paragraph("甲"), 10), (paragraph("乙"), 10)])

        with pytest.raises(ChildSplitError, match="数量"):
            chunk_parent(blocks, [10], target=100)
