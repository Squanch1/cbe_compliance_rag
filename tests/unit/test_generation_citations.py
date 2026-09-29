"""引用校验的单元测试。

这一层的价值全在「模型写错编号时不放过」上，所以边界要测到：越界的、
重复的、全角的、压根没写的。

它做不到的事也写在这里——判断不了引用是否恰当，那是评测集的活。
"""

from __future__ import annotations

from typing import Any

from cbe_rag.generation.citations import check_citations, parse_citations
from cbe_rag.retrieval.models import RetrievedParent


def make_parent(number: int, **overrides: Any) -> RetrievedParent:
    """造一份材料。"""
    fields: dict[str, Any] = {
        "parent_id": "doc-%d_p0000" % number,
        "doc_id": "doc-%d" % number,
        "score": 0.7,
        "text": "材料正文 %d" % number,
        "token_count": 20,
        "title": "材料 %d" % number,
        "source_url": "https://example.org/%d" % number,
        "country": "EU",
        "doc_type": "faq",
        "publisher": "amazon",
        "effective_date": None,
    }
    fields.update(overrides)
    return RetrievedParent(**fields)


def make_parents(count: int) -> list[RetrievedParent]:
    """造 count 份材料。"""
    return [make_parent(index) for index in range(1, count + 1)]


class TestParseCitations:
    def test_single_citation(self) -> None:
        assert parse_citations("这是结论 [1]。") == [1]

    def test_multiple_citations_in_order(self) -> None:
        assert parse_citations("结论 [2]，另有 [1]。") == [2, 1]

    def test_repeated_number_is_kept_once(self) -> None:
        # 同一份材料被引多次是常事，去重后输出更干净
        assert parse_citations("先 [1]，再 [1]，还是 [1]。") == [1]

    def test_adjacent_citations(self) -> None:
        # 提示词要求写 [1][2] 而不是 [1,2]
        assert parse_citations("结论 [1][2]。") == [1, 2]

    def test_multi_digit_number(self) -> None:
        assert parse_citations("见 [12]。") == [12]

    def test_full_width_brackets(self) -> None:
        # 中文回答里偶尔写成全角。只认半角的话，那种引用会被当成
        # 「没有引用」，判反了比不判更糟。
        assert parse_citations("结论 ［1］。") == [1]

    def test_no_citation(self) -> None:
        assert parse_citations("没有任何引用的一段话。") == []

    def test_plain_brackets_are_not_citations(self) -> None:
        # 方括号里不是数字就不算
        assert parse_citations("见 [注] 与 [a]。") == []


class TestCheckCitations:
    def test_valid_citations_map_to_parents(self) -> None:
        report = check_citations("结论 [1][2]。", make_parents(3))

        assert [parent.parent_id for parent in report.cited] == [
            "doc-1_p0000",
            "doc-2_p0000",
        ]

    def test_has_any_when_cited(self) -> None:
        assert check_citations("结论 [1]。", make_parents(2)).has_any

    def test_no_citation_means_nothing_cited(self) -> None:
        # 一条引用都没有的回答要被降级为「依据不足」
        report = check_citations("这是一段没有出处的结论。", make_parents(2))

        assert report.cited == []
        assert not report.has_any

    def test_out_of_range_number_is_invalid(self) -> None:
        # 编号越界是模型幻觉的常见形式。不拦住的话，一个 [7] 会被原样
        # 呈现给使用者，看着像有出处。
        report = check_citations("结论 [7]。", make_parents(2))

        assert report.invalid_numbers == [7]
        assert report.cited == []

    def test_zero_is_out_of_range(self) -> None:
        # 编号从 1 开始
        report = check_citations("结论 [0]。", make_parents(2))

        assert report.invalid_numbers == [0]

    def test_mixed_valid_and_invalid(self) -> None:
        report = check_citations("先 [1]，再 [9]。", make_parents(2))

        assert [parent.parent_id for parent in report.cited] == ["doc-1_p0000"]
        assert report.invalid_numbers == [9]

    def test_cited_is_sorted_by_material_order(self) -> None:
        # 按引用出现的顺序输出会让同一份材料在两次运行里位置不同
        report = check_citations("先 [3]，再 [1]。", make_parents(3))

        assert [parent.parent_id for parent in report.cited] == [
            "doc-1_p0000",
            "doc-3_p0000",
        ]

    def test_empty_material_makes_every_citation_invalid(self) -> None:
        report = check_citations("结论 [1]。", [])

        assert report.invalid_numbers == [1]
        assert not report.has_any

    def test_repeated_citation_lists_the_material_once(self) -> None:
        report = check_citations("先 [1]，再 [1]。", make_parents(2))

        assert len(report.cited) == 1
