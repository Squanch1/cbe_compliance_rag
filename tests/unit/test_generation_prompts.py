"""提示词模板的单元测试。

模板本身没什么可测的，值得测的是两件容易出错的事：

- 编号与材料的对应关系。回答里的 `[n]` 靠它映射回 parent_id，编号错了
  或断了，引用就会指向另一份材料——而且不报错。
- 未标注生效日期时是否如实写出来。留空会让模型以为这一栏被省略了，
  而它必须知道「这份材料没有日期」才会按规则说明。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cbe_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
    has_unmarked_effective_date,
)
from cbe_rag.retrieval.models import RetrievedParent


def make_parent(**overrides: Any) -> RetrievedParent:
    """造一条材料。"""
    fields: dict[str, Any] = {
        "parent_id": "doc-1_p0000",
        "doc_id": "doc-1",
        "score": 0.7,
        "text": "进口一站式服务适用于价值不超过 150 欧元的货物。",
        "token_count": 20,
        "title": "欧洲增值税常见问题",
        "source_url": "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX",
        "country": "EU",
        "doc_type": "policy",
        "publisher": "amazon",
        "effective_date": None,
    }
    fields.update(overrides)
    return RetrievedParent(**fields)


class TestSystemPrompt:
    def test_states_the_citation_rule(self) -> None:
        assert "[1]" in SYSTEM_PROMPT

    def test_forbids_answering_beyond_the_material(self) -> None:
        # 不编造是这个项目的底线
        assert "不要" in SYSTEM_PROMPT

    def test_requires_declaring_insufficient_evidence(self) -> None:
        assert "依据不足" in SYSTEM_PROMPT

    def test_states_the_recency_rule(self) -> None:
        # 多份规定冲突时以生效日期最新者为准（01-scope 3.4）
        assert "生效日期最新" in SYSTEM_PROMPT

    def test_forbids_inventing_an_effective_date(self) -> None:
        # 02-architecture 6.2.2：不得推断一个日期填上
        assert "不得推断" in SYSTEM_PROMPT


class TestBuildContext:
    def test_numbers_start_at_one(self) -> None:
        context = build_context([make_parent()])

        assert context.startswith("[1] ")

    def test_numbers_are_continuous(self) -> None:
        # 断号会让模型跳过缺失的编号，也让校验多一层分支
        context = build_context([make_parent(), make_parent(), make_parent()])

        assert "[1] " in context
        assert "[2] " in context
        assert "[3] " in context

    def test_each_material_carries_its_metadata(self) -> None:
        context = build_context([make_parent()])

        assert "amazon" in context
        assert "EU" in context
        assert "policy" in context

    def test_carries_the_source_url(self) -> None:
        # 引用要能回溯到原文
        context = build_context([make_parent()])

        assert "sellercentral.amazon.com" in context

    def test_marks_a_missing_effective_date(self) -> None:
        # 留空会让模型以为这一栏被省略了
        context = build_context([make_parent(effective_date=None)])

        assert "生效日期：未标注" in context

    def test_writes_the_effective_date_when_present(self) -> None:
        context = build_context([make_parent(effective_date=date(2021, 7, 1))])

        assert "生效日期：2021-07-01" in context

    def test_marks_a_missing_source_url(self) -> None:
        context = build_context([make_parent(source_url=None)])

        assert "未登记" in context

    def test_keeps_the_full_text(self) -> None:
        # 材料正文是要送进模型的部分，不能被截断
        long_text = "字" * 3000

        context = build_context([make_parent(text=long_text)])

        assert long_text in context

    def test_empty_list_says_so(self) -> None:
        # 空材料不能让模型以为「没有限制」
        assert "没有检索到任何材料" in build_context([])


class TestBuildUserPrompt:
    def test_puts_the_question_first(self) -> None:
        # 材料动辄上万 token，问题放后面会被推得很远
        prompt = build_user_prompt("进口一站式服务的上限是多少", [make_parent()])

        assert prompt.startswith("问题：进口一站式服务的上限是多少")

    def test_includes_the_context(self) -> None:
        prompt = build_user_prompt("问题", [make_parent()])

        assert "[1] " in prompt

    def test_works_without_material(self) -> None:
        prompt = build_user_prompt("问题", [])

        assert "没有检索到任何材料" in prompt


class TestHasUnmarkedEffectiveDate:
    def test_detects_a_missing_date(self) -> None:
        assert has_unmarked_effective_date([make_parent(effective_date=None)])

    def test_false_when_all_dated(self) -> None:
        parents = [
            make_parent(effective_date=date(2021, 7, 1)),
            make_parent(effective_date=date(2027, 1, 1)),
        ]

        assert not has_unmarked_effective_date(parents)

    def test_true_when_only_one_is_missing(self) -> None:
        # 一份没标就够触发提示了
        parents = [
            make_parent(effective_date=date(2021, 7, 1)),
            make_parent(effective_date=None),
        ]

        assert has_unmarked_effective_date(parents)

    def test_empty_list_has_nothing_unmarked(self) -> None:
        assert not has_unmarked_effective_date([])
