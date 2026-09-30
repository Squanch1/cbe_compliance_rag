"""阈值标定的单元测试。

只测纯函数：候选阈值的算法与报告渲染。脚本里连服务、跑模型的那些部分
不在这里测——它们是运维入口。
"""

from __future__ import annotations

from cbe_rag.evaluation import EXPECTED_ANSWER, EXPECTED_REFUSE, EvalCase

from calibrate_threshold import CaseScore, render, suggest_thresholds


def make_score(score: float | None, *, should_answer: bool) -> CaseScore:
    """造一条用例的检索结果。"""
    return CaseScore(
        case=EvalCase(
            line_number=2,
            question="问题",
            expected_behavior=EXPECTED_ANSWER if should_answer else EXPECTED_REFUSE,
            category="测试",
            expected_sources=(),
            notes=None,
        ),
        score=score,
        elapsed_ms=1.0,
    )


class TestSuggestThresholds:
    def test_blocks_the_irrelevant_ones(self) -> None:
        scores = [
            make_score(0.7, should_answer=True),
            make_score(0.3, should_answer=False),
        ]

        suggestions = suggest_thresholds(scores)

        assert suggestions[0].blocked == 1
        assert suggestions[0].missed == 0

    def test_never_hurts_a_should_answer_case(self) -> None:
        # 约束是不误伤：漏拦只是多调一次模型，误伤是用户拿不到能答的问题
        scores = [
            make_score(0.5, should_answer=True),
            make_score(0.4, should_answer=False),
        ]

        for item in suggest_thresholds(scores):
            assert item.missed == 0
            assert item.threshold <= 0.5

    def test_missing_score_is_blocked(self) -> None:
        # 一条都没检索到按最低分处理，一定会被拦下——这是对的，
        # 那种情况确实无关
        scores = [
            make_score(None, should_answer=False),
            make_score(0.7, should_answer=True),
        ]

        assert suggest_thresholds(scores)[0].blocked == 1

    def test_missing_score_never_blocks_a_should_answer_case(self) -> None:
        # 该答的却一条都没检索到，拦下它的线就是误伤，不该被推荐
        scores = [
            make_score(None, should_answer=True),
            make_score(0.3, should_answer=False),
        ]

        for item in suggest_thresholds(scores):
            assert item.missed == 0

    def test_lists_every_line_not_just_the_best(self) -> None:
        # 只给最优那条会让人以为它稳当——它往往贴着「该答的最低分」，
        # 一点余量都没有。各级线都要摆出来。
        scores = [
            make_score(0.8, should_answer=True),
            make_score(0.6, should_answer=False),
            make_score(0.3, should_answer=False),
        ]

        suggestions = suggest_thresholds(scores)

        assert len(suggestions) == 3
        assert [item.blocked for item in suggestions] == [2, 1, 0]

    def test_prefers_the_line_that_blocks_most(self) -> None:
        scores = [
            make_score(0.8, should_answer=True),
            make_score(0.6, should_answer=False),
            make_score(0.3, should_answer=False),
        ]

        suggestions = suggest_thresholds(scores)

        assert suggestions[0].threshold == 0.8
        assert suggestions[0].blocked == 2

    def test_never_suggests_a_line_that_hurts(self) -> None:
        # 该答的分数比该拒答的还低时，能拦住的线都会误伤。
        # 最低那条线不拦任何东西，因此它会被返回——它是唯一无害的，
        # 但也拦不住什么，得从 blocked 为 0 看出来。
        scores = [
            make_score(0.3, should_answer=True),
            make_score(0.6, should_answer=False),
        ]

        suggestions = suggest_thresholds(scores)

        assert suggestions
        assert all(item.missed == 0 for item in suggestions)
        assert suggestions[0].blocked == 0

    def test_empty_input(self) -> None:
        assert suggest_thresholds([]) == []


class TestRender:
    def make_scores(self) -> list[CaseScore]:
        return [
            make_score(0.74, should_answer=True),
            make_score(0.59, should_answer=True),
            make_score(0.31, should_answer=False),
        ]

    def test_lists_every_case(self) -> None:
        text = "\n".join(render(self.make_scores(), suggest_thresholds(self.make_scores())))

        assert "该答的" in text
        assert "该拒答的" in text

    def test_reports_the_counts(self) -> None:
        text = "\n".join(render(self.make_scores(), suggest_thresholds(self.make_scores())))

        assert "该答 2" in text
        assert "该拒答 1" in text

    def test_says_when_the_two_kinds_are_separable(self) -> None:
        text = "\n".join(render(self.make_scores(), suggest_thresholds(self.make_scores())))

        assert "全部高于" in text

    def test_flags_when_the_two_kinds_cannot_be_separated(self) -> None:
        # 该拒答的那条分数反而更高，怎么定阈值都要牺牲一条
        scores = [
            make_score(0.50, should_answer=True),
            make_score(0.55, should_answer=False),
        ]

        text = "\n".join(render(scores, suggest_thresholds(scores)))

        assert "分不开" in text
        assert "都会被判错" in text

    def test_shows_a_missing_score_as_such(self) -> None:
        scores = self.make_scores() + [make_score(None, should_answer=False)]

        text = "\n".join(render(scores, suggest_thresholds(scores)))

        assert "无命中" in text

    def test_explains_what_the_line_is_for(self) -> None:
        # 别让人以为这条线能判「材料够不够回答」
        text = "\n".join(render(self.make_scores(), suggest_thresholds(self.make_scores())))

        assert "交给模型" in text

    def test_says_why_the_two_kinds_cannot_be_fully_separated(self) -> None:
        # 这条结论是从实测数据得出的，写在报告里免得下次又去试别的分值
        text = "\n".join(render(self.make_scores(), suggest_thresholds(self.make_scores())))

        assert "分不出" in text

    def test_empty_set_says_so(self) -> None:
        text = "\n".join(render([], []))

        assert "没法定阈值" in text


class TestRenderMargin:
    """余量列的语义：离「该答的最低分」还有多远，越大越安全。

    粗筛线只能定在该答最低分**之下**（定在上面就误伤），所以余量必然是
    零或正数。符号写反的话，最危险的那条线（贴着该答最低分）会显示成
    余量最小，最安全的那条反而显示成负数——读的人正好会看反。
    """

    def make_scores(self) -> list[CaseScore]:
        return [
            make_score(0.5867, should_answer=True),
            make_score(0.4877, should_answer=False),
        ]

    def test_a_safer_line_has_a_bigger_margin(self) -> None:
        # 0.4877 这条拦不住东西，但它离该答最低分最远，余量应当最大
        scores = self.make_scores()

        text = "\n".join(render(scores, suggest_thresholds(scores)))

        assert "余量 0.0990" in text

    def test_no_line_shows_a_negative_margin(self) -> None:
        scores = self.make_scores()

        text = "\n".join(render(scores, suggest_thresholds(scores)))

        assert "-0.0990" not in text

    def test_the_line_at_the_lowest_answer_has_no_margin(self) -> None:
        scores = self.make_scores()

        text = "\n".join(render(scores, suggest_thresholds(scores)))

        assert "余量 0.0000" in text
