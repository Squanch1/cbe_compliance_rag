"""评测集读取的单元测试。

用临时文件造数据，不读仓库里那份评测集——那份是给人改的，测试跟着它变
会一直红。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cbe_rag.evaluation.cases import (
    EXPECTED_ANSWER,
    EXPECTED_REFUSE,
    EvalCaseError,
    load_eval_cases,
    summarize,
)

HEADER = ["question", "expected_behavior", "category", "expected_sources", "notes"]


def write_cases(
    tmp_path: Path, *rows: dict[str, str], encoding: str = "utf-8"
) -> Path:
    """把若干行写成一份评测集 CSV。"""
    path = tmp_path / "eval_cases.csv"
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in HEADER})
    return path


def case(**overrides: str) -> dict[str, str]:
    """造一行评测用例，字段可按需覆盖。"""
    row = {
        "question": "进口一站式服务的适用金额上限是多少",
        "expected_behavior": EXPECTED_ANSWER,
        "category": "税种规则",
        "expected_sources": "eu-vat-ecommerce-explanatory-notes-rev-2027.pdf",
        "notes": "",
    }
    row.update(overrides)
    return row


class TestLoading:
    def test_reads_a_case(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case())

        cases = load_eval_cases(path)

        assert len(cases) == 1
        assert cases[0].question == "进口一站式服务的适用金额上限是多少"

    def test_records_the_line_number(self, tmp_path: Path) -> None:
        # 表头占第 1 行，数据从第 2 行起
        path = write_cases(tmp_path, case(), case(question="第二个问题"))

        cases = load_eval_cases(path)

        assert [item.line_number for item in cases] == [2, 3]

    def test_tolerates_a_bom(self, tmp_path: Path) -> None:
        # Excel 存 UTF-8 CSV 会带 BOM。不处理的话它会并进第一列的列名里，
        # 整张表都认不出 question。
        path = write_cases(tmp_path, case(), encoding="utf-8-sig")

        cases = load_eval_cases(path)

        assert len(cases) == 1

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EvalCaseError, match="不存在"):
            load_eval_cases(tmp_path / "absent.csv")

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "eval_cases.csv"
        path.write_text("question,expected_behavior\n问题,answer\n", encoding="utf-8")

        with pytest.raises(EvalCaseError, match="表头缺少列"):
            load_eval_cases(path)


class TestValidation:
    def test_empty_question_raises_with_line_number(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(question=""))

        with pytest.raises(EvalCaseError, match="第 2 行"):
            load_eval_cases(path)

    def test_unknown_behavior_raises(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_behavior="maybe"))

        with pytest.raises(EvalCaseError, match="不在允许范围内"):
            load_eval_cases(path)

    def test_empty_behavior_raises(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_behavior=""))

        with pytest.raises(EvalCaseError, match="expected_behavior 为空"):
            load_eval_cases(path)

    def test_empty_category_raises(self, tmp_path: Path) -> None:
        # 分类是给人看统计的，空着等于没分
        path = write_cases(tmp_path, case(category=""))

        with pytest.raises(EvalCaseError, match="category 为空"):
            load_eval_cases(path)


class TestSources:
    def test_single_source(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_sources="a.pdf"))

        assert load_eval_cases(path)[0].expected_sources == ("a.pdf",)

    def test_multiple_sources_split_on_semicolon(self, tmp_path: Path) -> None:
        # 用分号不用逗号：CSV 里逗号要转义，人填的时候容易忘
        path = write_cases(tmp_path, case(expected_sources="a.pdf; b.pdf"))

        assert load_eval_cases(path)[0].expected_sources == ("a.pdf", "b.pdf")

    def test_empty_sources_yield_empty_tuple(self, tmp_path: Path) -> None:
        # 拒答类通常不写期望来源
        path = write_cases(tmp_path, case(expected_sources="", expected_behavior=EXPECTED_REFUSE))

        assert load_eval_cases(path)[0].expected_sources == ()

    def test_blank_entries_are_dropped(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_sources="a.pdf;;  ;b.pdf"))

        assert load_eval_cases(path)[0].expected_sources == ("a.pdf", "b.pdf")


class TestShouldAnswer:
    def test_answer_case(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_behavior=EXPECTED_ANSWER))

        assert load_eval_cases(path)[0].should_answer is True

    def test_refuse_case(self, tmp_path: Path) -> None:
        path = write_cases(tmp_path, case(expected_behavior=EXPECTED_REFUSE))

        assert load_eval_cases(path)[0].should_answer is False


class TestSummarize:
    def test_counts_both_kinds(self, tmp_path: Path) -> None:
        path = write_cases(
            tmp_path,
            case(expected_behavior=EXPECTED_ANSWER),
            case(expected_behavior=EXPECTED_ANSWER),
            case(expected_behavior=EXPECTED_REFUSE),
        )

        summary = summarize(load_eval_cases(path))

        assert summary.answer_count == 2
        assert summary.refuse_count == 1
        assert summary.total == 3

    def test_empty_set(self) -> None:
        summary = summarize([])

        assert summary.total == 0
