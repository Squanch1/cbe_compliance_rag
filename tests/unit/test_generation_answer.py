"""回答生成的单元测试。

模型用假实现，不发起真实请求。

降级这条路最关键：一段没有出处的结论必须被换掉，而不是原样返回——
使用者看不出它没有依据，这正是「禁止编造」最容易被绕过的地方。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cbe_rag.generation.answer import NO_EVIDENCE_TEXT, generate_answer
from cbe_rag.retrieval.models import RetrievedParent
from cbe_rag.storage.bailian_client import Completion


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
        "effective_date": date(2021, 7, 1),
    }
    fields.update(overrides)
    return RetrievedParent(**fields)


def make_parents(count: int, **overrides: Any) -> list[RetrievedParent]:
    """造 count 份材料。"""
    return [
        make_parent(index, **overrides) for index in range(1, count + 1)
    ]


class FakeClient:
    """假的模型客户端，返回预设文本并记录收到的消息。"""

    def __init__(self, text: str = "回答 [1]。") -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(text=self._text, model="fake")


class TestSuccessfulAnswer:
    def test_returns_the_model_text(self) -> None:
        client = FakeClient("进口一站式服务适用于 150 欧元以下 [1]。")

        answer = generate_answer("上限是多少", make_parents(1), client)

        assert answer.text.startswith("进口一站式服务适用于")

    def test_not_degraded_when_cited(self) -> None:
        answer = generate_answer("问题", make_parents(2), FakeClient("结论 [1]。"))

        assert answer.degraded is False

    def test_resolves_the_citation(self) -> None:
        answer = generate_answer("问题", make_parents(2), FakeClient("结论 [2]。"))

        assert [parent.parent_id for parent in answer.citations.cited] == [
            "doc-2_p0000"
        ]

    def test_sends_the_system_prompt(self) -> None:
        client = FakeClient()

        generate_answer("问题", make_parents(1), client)

        system, _ = client.calls[0]
        assert "[1]" in system  # 引用规则在系统提示词里

    def test_sends_the_question_and_material(self) -> None:
        client = FakeClient()

        generate_answer("上限是多少", make_parents(1), client)

        _, user = client.calls[0]
        assert "上限是多少" in user
        assert "材料 1" in user

    def test_raw_text_matches_the_model_output(self) -> None:
        answer = generate_answer("问题", make_parents(1), FakeClient("原文 [1]。"))

        assert answer.raw_text == "原文 [1]。"


class TestDegradedAnswer:
    def test_answer_without_citation_is_replaced(self) -> None:
        # 使用者看不出「没有出处的结论」和「有出处的结论」的区别，
        # 所以必须由程序换掉
        client = FakeClient("这是一段没有出处的结论。")

        answer = generate_answer("问题", make_parents(1), client)

        assert answer.text == NO_EVIDENCE_TEXT
        assert answer.degraded is True

    def test_raw_text_keeps_what_the_model_wrote(self) -> None:
        # 降级时最想知道的就是它到底写了什么
        client = FakeClient("这是一段没有出处的结论。")

        answer = generate_answer("问题", make_parents(1), client)

        assert answer.raw_text == "这是一段没有出处的结论。"

    def test_keeps_the_citation_report(self) -> None:
        answer = generate_answer("问题", make_parents(1), FakeClient("没有引用。"))

        assert answer.citations.cited == []
        assert not answer.citations.has_any

    def test_no_material_skips_the_model(self) -> None:
        # 调用只会得到一段没有依据的话，然后被拦下——白花钱
        client = FakeClient()

        answer = generate_answer("问题", [], client)

        assert answer.text == NO_EVIDENCE_TEXT
        assert answer.degraded is True
        assert client.calls == []

    def test_no_material_has_no_raw_text(self) -> None:
        answer = generate_answer("问题", [], FakeClient())

        assert answer.raw_text == ""


class TestInvalidCitations:
    def test_notes_the_invalid_number(self) -> None:
        # 模型编了一个来源，意味着它对这一段把握值得怀疑，要说出来
        client = FakeClient("结论 [1]，另有 [9]。")

        answer = generate_answer("问题", make_parents(2), client)

        assert any("9" in note for note in answer.notes)

    def test_valid_citations_survive(self) -> None:
        client = FakeClient("结论 [1]，另有 [9]。")

        answer = generate_answer("问题", make_parents(2), client)

        assert answer.degraded is False
        assert [parent.parent_id for parent in answer.citations.cited] == [
            "doc-1_p0000"
        ]

    def test_all_invalid_degrades(self) -> None:
        # 全都越界等于没有引用
        client = FakeClient("结论 [8][9]。")

        answer = generate_answer("问题", make_parents(2), client)

        assert answer.degraded is True


class TestEffectiveDateNotes:
    def test_notes_unmarked_effective_date(self) -> None:
        # 02-architecture 6.2.3：必须显式告知无法判断孰新
        parents = make_parents(2)
        parents[1] = make_parent(2, effective_date=None)

        answer = generate_answer("问题", parents, FakeClient("结论 [1]。"))

        assert any("生效日期" in note for note in answer.notes)

    def test_no_note_when_all_dated(self) -> None:
        answer = generate_answer("问题", make_parents(2), FakeClient("结论 [1]。"))

        assert answer.notes == []

    def test_note_reaches_a_degraded_answer_too(self) -> None:
        # 降级不代表时效问题就不存在了，该说的还要说
        parents = [make_parent(1, effective_date=None)]

        answer = generate_answer("问题", parents, FakeClient("没有引用。"))

        assert answer.degraded is True
        assert any("生效日期" in note for note in answer.notes)
