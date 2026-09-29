"""父块切分的单元测试。"""

from __future__ import annotations

import pytest

from cbe_rag.ingestion.chunker.parent import (
    ParentSplitError,
    split_parents,
)
from cbe_rag.ingestion.parser.schema import Block, BlockType


def heading(text: str) -> Block:
    return Block(type=BlockType.HEADING, text=text, order=0, level=1)


def paragraph(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text, order=0)


def blocks_and_counts(specs: list[tuple[Block, int]]) -> tuple[list[Block], list[int]]:
    return [b for b, _ in specs], [c for _, c in specs]


CONST = dict(lower=60, upper=150)


class TestBasicGrouping:
    def test_empty_input_yields_no_parents(self) -> None:
        assert split_parents([], [], **CONST) == []

    def test_small_document_becomes_one_parent(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲"), 20), (paragraph("乙"), 20)])

        parents = split_parents(blocks, counts, **CONST)

        assert len(parents) == 1
        assert len(parents[0]) == 2

    def test_all_blocks_are_kept(self) -> None:
        # 切分只分组，不丢内容
        blocks, counts = blocks_and_counts([(paragraph(str(i)), 20) for i in range(20)])

        parents = split_parents(blocks, counts, **CONST)

        assert sum(len(p) for p in parents) == 20


class TestHeadingIsPreferredBreak:
    def test_breaks_before_heading_once_lower_bound_reached(self) -> None:
        blocks, counts = blocks_and_counts([
            (paragraph("段一"), 40),
            (paragraph("段二"), 40),   # 累积 80 >= 60，够了下限
            (heading("标题"), 10),     # ← 在这里断
            (paragraph("段三"), 40),
        ])

        parents = split_parents(blocks, counts, **CONST)

        assert [len(p) for p in parents] == [2, 2]
        assert parents[1][0].type is BlockType.HEADING

    def test_does_not_break_before_heading_when_below_lower_bound(self) -> None:
        # 累积不够就断会产生大量碎片父块——实测真实文档里 66% 的
        # 「按标题切」结果都不到 300 token，正是这个原因
        blocks, counts = blocks_and_counts([
            (paragraph("短"), 10),
            (heading("标题"), 10),     # 累积才 10，不够下限，不断
            (paragraph("内容"), 40),
        ])

        parents = split_parents(blocks, counts, **CONST)

        assert len(parents) == 1

    def test_heading_starts_the_new_parent(self) -> None:
        blocks, counts = blocks_and_counts([
            (paragraph("前"), 70),
            (heading("新章节"), 10),
            (paragraph("后"), 10),
        ])

        parents = split_parents(blocks, counts, **CONST)

        assert parents[1][0].text == "新章节"


class TestUpperBoundForcesBreak:
    def test_breaks_at_upper_bound_without_heading(self) -> None:
        # 找不到标题也要断，否则一个超长章节会得到远超模型上限的父块。
        # 实测欧盟文档里最大的章节有 13007 token，超过模型的 8192。
        blocks, counts = blocks_and_counts([(paragraph(str(i)), 50) for i in range(6)])

        parents = split_parents(blocks, counts, **CONST)

        assert len(parents) > 1
        # 每块 50 token，上限 150，因此最多 3 个块
        assert all(len(p) <= 3 for p in parents)

    def test_upper_bound_is_soft(self) -> None:
        # 切分只能发生在块与块之间，因此实际大小会略微超过上限。
        # 这不是缺陷：超出量不超过单个最大块，对能否喂进模型没有影响。
        # 改成「加块之前判断」会让上限真正生效，但会留下孤儿块。
        blocks, counts = blocks_and_counts(
            [(paragraph("甲"), 80), (paragraph("乙"), 80), (paragraph("丙"), 80)]
        )

        parents = split_parents(blocks, counts, lower=60, upper=100)

        # 累积到 80 未达上限，再加一块得 160 才断
        assert len(parents[0]) == 2

    def test_single_oversized_block_becomes_its_own_parent(self) -> None:
        # 单个块超过上限时切不开——块内没有更小的边界可用。
        # 这里是记录这个限制，不是理想结果。
        blocks, counts = blocks_and_counts([(paragraph("巨块"), 500), (paragraph("小"), 10)])

        parents = split_parents(blocks, counts, **CONST)

        assert parents[0] == [blocks[0]]


class TestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        # counts 与 blocks 对不上说明调用方算错了，是 bug 不是数据问题
        blocks, _ = blocks_and_counts([(paragraph("甲"), 10), (paragraph("乙"), 10)])

        with pytest.raises(ParentSplitError, match="数量"):
            split_parents(blocks, [10], **CONST)

    def test_lower_bound_above_upper_bound_raises(self) -> None:
        blocks, counts = blocks_and_counts([(paragraph("甲"), 10)])

        with pytest.raises(ParentSplitError, match="下限"):
            split_parents(blocks, counts, lower=200, upper=150)
