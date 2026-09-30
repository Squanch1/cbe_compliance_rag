"""FAQ 翻译产物解析的单元测试。

只测纯函数 parse_translation。读写文件的那部分是运维入口，不在这里测。
"""

from __future__ import annotations

import pytest

from convert_faq_translation import TranslationError, parse_translation


def make_entry(
    entry_id: str = "F-01",
    *,
    question: str = "用 IOSS 发货给欧盟买家，对我有什么好处？",
    answer_zh: str = "使用 IOSS 可以让买家在购买时支付含税价，不必在进口时再付一笔。",
    answer_en: str = "By using the IOSS, a supplier ensures a transparent transaction.",
    source: str = "欧盟指南 4.2.10 (a) 问题 1",
) -> str:
    """造一条格式正确的译文。"""
    return "\n".join(
        [
            "=== %s ===" % entry_id,
            "--- 问题 ---",
            question,
            "--- 译文 ---",
            answer_zh,
            "--- 原文 ---",
            answer_en,
            "--- 出处 ---",
            source,
        ]
    )


class TestParseTranslation:
    def test_parses_a_single_entry(self) -> None:
        entries = parse_translation(make_entry())

        assert len(entries) == 1
        assert entries[0].entry_id == "F-01"
        assert entries[0].question_zh == "用 IOSS 发货给欧盟买家，对我有什么好处？"
        assert entries[0].answer_en.startswith("By using the IOSS")
        assert entries[0].source == "欧盟指南 4.2.10 (a) 问题 1"

    def test_parses_several_entries_in_order(self) -> None:
        markdown = make_entry("F-01") + "\n\n" + make_entry("F-02") + "\n\n" + make_entry("F-03")

        entries = parse_translation(markdown)

        assert [item.entry_id for item in entries] == ["F-01", "F-02", "F-03"]

    def test_keeps_multiple_paragraphs(self) -> None:
        # 答案本来就是多段的，空行必须留住，不能压成一行
        answer = "第一段。\n\n第二段。"
        entries = parse_translation(make_entry(answer_zh=answer))

        assert entries[0].answer_zh == answer

    def test_tolerates_extra_dashes_and_spaces(self) -> None:
        # 模型多画两根横线、多敲几个空格，不该让整批返工
        markdown = "\n".join(
            [
                "=====   F-01  =====",
                "-- 问题 --",
                "问题内容？",
                "-- 译文 --",
                "译文内容。",
                "-- 原文 --",
                "Some English text.",
                "-- 出处 --",
                "某处",
            ]
        )

        entries = parse_translation(markdown)

        assert entries[0].entry_id == "F-01"
        assert entries[0].answer_en == "Some English text."

    def test_ignores_code_fence_lines(self) -> None:
        markdown = "```\n" + make_entry() + "\n```\n"

        entries = parse_translation(markdown)

        assert len(entries) == 1
        assert entries[0].answer_zh.endswith("不必在进口时再付一笔。")

    def test_ignores_text_before_the_first_entry(self) -> None:
        # 模型常会先来一句「好的，以下是译文」，不该被当成正文
        markdown = "好的，以下是译文：\n\n" + make_entry()

        entries = parse_translation(markdown)

        assert entries[0].question_zh.startswith("用 IOSS")

    def test_keeps_prose_that_looks_like_a_separator(self) -> None:
        # 正文里出现 `=== 注意 ===` 不该把条目劈成两半——分隔线必须带 F-编号
        markdown = "\n".join(
            [
                "=== F-01 ===",
                "--- 问题 ---",
                "问题内容？",
                "--- 译文 ---",
                "第一段。",
                "",
                "=== 注意 ===",
                "",
                "第二段。",
                "--- 原文 ---",
                "Some English text.",
                "--- 出处 ---",
                "某处",
            ]
        )

        entries = parse_translation(markdown)

        assert len(entries) == 1
        assert "=== 注意 ===" in entries[0].answer_zh
        assert "第二段。" in entries[0].answer_zh

    def test_rejects_a_missing_field(self) -> None:
        markdown = "\n".join(
            [
                "=== F-01 ===",
                "--- 问题 ---",
                "问题内容？",
                "--- 原文 ---",
                "Some English text.",
                "--- 出处 ---",
                "某处",
            ]
        )

        with pytest.raises(TranslationError) as excinfo:
            parse_translation(markdown)

        assert "译文" in str(excinfo.value)

    def test_rejects_an_empty_field(self) -> None:
        markdown = "\n".join(
            [
                "=== F-01 ===",
                "--- 问题 ---",
                "问题内容？",
                "--- 译文 ---",
                "--- 原文 ---",
                "Some English text.",
                "--- 出处 ---",
                "某处",
            ]
        )

        with pytest.raises(TranslationError) as excinfo:
            parse_translation(markdown)

        assert "译文" in str(excinfo.value)
        assert "第 1 行" in str(excinfo.value)

    def test_rejects_a_duplicate_id(self) -> None:
        first = make_entry("F-01")
        markdown = first + "\n\n" + make_entry("F-01")
        second_line = len(first.splitlines()) + 2

        with pytest.raises(TranslationError) as excinfo:
            parse_translation(markdown)

        assert "F-01" in str(excinfo.value)
        assert "第 %d 行" % second_line in str(excinfo.value)

    def test_rejects_a_source_without_latin_letters(self) -> None:
        # 模型没把英文原样回填，而是复制了中文——这种要拦下来
        with pytest.raises(TranslationError) as excinfo:
            parse_translation(make_entry(answer_en="这段是中文，不是原文。"))

        assert "原文" in str(excinfo.value)

    def test_rejects_a_field_marker_outside_any_entry(self) -> None:
        with pytest.raises(TranslationError) as excinfo:
            parse_translation("--- 译文 ---\n译文内容。\n")

        assert "不属于任何条目" in str(excinfo.value)

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(TranslationError) as excinfo:
            parse_translation("")

        assert "没有解析出任何条目" in str(excinfo.value)

    def test_rejects_prose_only_input(self) -> None:
        with pytest.raises(TranslationError):
            parse_translation("这是模型没按格式输出的一整段话。\n")
