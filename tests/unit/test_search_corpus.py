"""检索演示脚本的单元测试。

只测渲染部分——它是纯函数，而且是使用者唯一会看的东西：分数、出处、
生效日期、正文片段各占一行，漏掉哪一行都会让这个脚本失去意义。

脚本其余部分依赖真实服务，不在这里测。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.models import RetrievalQuery, RetrievedParent

from search_corpus import describe_filters, preview, render_parents, render_verdict

TODAY = date(2026, 9, 29)


def make_parent(**overrides: Any) -> RetrievedParent:
    """造一条恢复出来的父块。"""
    fields: dict[str, Any] = {
        "parent_id": "doc-1_p0000",
        "doc_id": "doc-1",
        "score": 0.697,
        "text": "进口一站式服务适用于价值不超过 150 欧元的货物。",
        "title": "欧洲增值税常见问题",
        "source_url": "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX",
        "country": "EU",
        "doc_type": "faq",
        "publisher": "amazon",
        "effective_date": None,
    }
    fields.update(overrides)
    return RetrievedParent(**fields)


class TestPreview:
    def test_short_text_stays_on_one_line(self) -> None:
        assert preview("很短的一句") == ["很短的一句"]

    def test_blank_lines_are_folded(self) -> None:
        # 父块正文里带着排版留下的换行，原样打出来会碎成一堆短行
        assert preview("第一行\n\n第二行") == ["第一行 第二行"]

    def test_long_text_is_truncated_with_a_total(self) -> None:
        lines = preview("字" * 500, limit=100)

        assert len(lines) == 2
        assert lines[0].endswith("……")
        assert "共 500 字" in lines[1]

    def test_empty_text_is_marked(self) -> None:
        assert preview("   ") == ["（空）"]


class TestRenderParents:
    def test_shows_the_score_and_title(self) -> None:
        text = "\n".join(render_parents([make_parent()]))

        assert "0.6970" in text
        assert "欧洲增值税常见问题" in text

    def test_shows_the_source_url(self) -> None:
        # 「可追溯」是这个项目的核心主张，出处必须打得出来
        text = "\n".join(render_parents([make_parent()]))

        assert "sellercentral.amazon.com" in text

    def test_marks_a_missing_source_url(self) -> None:
        # 空白比「（未登记）」更容易被当成渲染漏了
        text = "\n".join(render_parents([make_parent(source_url=None)]))

        assert "未登记" in text

    def test_marks_a_missing_effective_date(self) -> None:
        # 这类文档回答时必须写明「未标注生效日期」，这里先给它一个提示
        text = "\n".join(render_parents([make_parent(effective_date=None)]))

        assert "未标注" in text

    def test_shows_the_effective_date_when_present(self) -> None:
        parent = make_parent(effective_date=date(2021, 7, 1))

        text = "\n".join(render_parents([parent]))

        assert "2021-07-01" in text

    def test_shows_the_filter_dimensions(self) -> None:
        text = "\n".join(render_parents([make_parent()]))

        assert "EU / faq / amazon" in text

    def test_numbers_the_results_in_order(self) -> None:
        parents = [make_parent(score=0.9), make_parent(score=0.5)]

        lines = render_parents(parents)

        assert lines[0].startswith("  1.")
        # 第二条的起始行：第一条占了标题加四行属性加一行分隔加正文
        assert any(line.startswith("  2.") for line in lines)

    def test_empty_list_renders_nothing(self) -> None:
        assert render_parents([]) == []


class TestRenderVerdict:
    def test_uncalibrated_threshold_says_so(self) -> None:
        # 判不了就说判不了，不猜一个结果出来
        text = "\n".join(render_verdict(0.6, RetrievalConfig(refuse_threshold=None)))

        assert "未标定" in text

    def test_shows_the_criterion(self) -> None:
        text = "\n".join(render_verdict(0.5867, RetrievalConfig()))

        assert "0.5867" in text

    def test_sufficient_evidence(self) -> None:
        text = "\n".join(render_verdict(0.6, RetrievalConfig(refuse_threshold=0.45)))

        assert "证据够" in text

    def test_insufficient_evidence(self) -> None:
        text = "\n".join(render_verdict(0.3, RetrievalConfig(refuse_threshold=0.45)))

        assert "拒答" in text

    def test_no_hits_is_reported_as_such(self) -> None:
        text = "\n".join(render_verdict(None, RetrievalConfig()))

        assert "没有任何命中" in text

    def test_no_hits_is_insufficient_when_calibrated(self) -> None:
        # 一条都没检索到，阈值再低也不该判够
        text = "\n".join(render_verdict(None, RetrievalConfig(refuse_threshold=0.0)))

        assert "拒答" in text


class TestDescribeFilters:
    def test_no_filter(self) -> None:
        assert describe_filters(RetrievalQuery(text="q")) == "无"

    def test_single_filter(self) -> None:
        query = RetrievalQuery(text="q", country="EU")

        assert describe_filters(query) == "国家=EU"

    def test_multiple_filters(self) -> None:
        query = RetrievalQuery(text="q", country="EU", doc_type="guideline")

        assert describe_filters(query) == "国家=EU、类型=guideline"
