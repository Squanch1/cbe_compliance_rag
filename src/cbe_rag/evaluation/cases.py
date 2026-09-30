"""评测集的读取与校验。

评测集是手工维护的 CSV，一行一道题。本模块把它读成结构化用例，取值有问题
时报出**行号**——手工表格最怕「有一处不对但不知道在哪一行」，这是从
manifest.py 那里沿用下来的做法。

**期望行为必须人工判定。** 判据是「语料里到底有没有能回答这个问题的内容」，
只能人去读。绝不能先跑一遍看分数、再回头把分低的标成 refuse——那是用分数
定义类别，再拿类别去标定阈值，绕一圈等于什么也没定。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from cbe_rag.config.settings import PROJECT_ROOT

# 路径按项目根推算，不依赖当前工作目录（见 CLAUDE.md 5.2）
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval_cases.csv"

COLUMNS: tuple[str, ...] = (
    "question",
    "expected_behavior",
    "category",
    "expected_sources",
    "notes",
)

# 两种期望行为。answer 是「语料里有答案，应当回答」，refuse 是「语料里没有，
# 应当拒答」——后者才是标定阈值的下界来源。
EXPECTED_ANSWER = "answer"
EXPECTED_REFUSE = "refuse"
ALLOWED_BEHAVIORS: frozenset[str] = frozenset({EXPECTED_ANSWER, EXPECTED_REFUSE})

# 多个期望来源之间的分隔符。用分号而不是逗号：CSV 里逗号要转义，人填的时候
# 容易忘。
_SOURCE_SEPARATOR = ";"


class EvalCaseError(Exception):
    """评测集内容有问题。消息里带行号，便于定位。"""


@dataclass(frozen=True)
class EvalCase:
    """一道评测题。"""

    line_number: int
    question: str
    expected_behavior: str
    category: str
    expected_sources: tuple[str, ...]
    notes: str | None

    @property
    def should_answer(self) -> bool:
        """这道题期望回答还是拒答。"""
        return self.expected_behavior == EXPECTED_ANSWER


@dataclass(frozen=True)
class EvalSummary:
    """评测集的规模统计。

    两类各多少条要分开看：**只有「该答」的题目标不出阈值下界**，
    只有「该拒答」的标不出上界。
    """

    answer_count: int
    refuse_count: int

    @property
    def total(self) -> int:
        return self.answer_count + self.refuse_count


def _clean(raw: str | None) -> str | None:
    """去掉首尾空白；空字符串视为未填写。"""
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def _parse_sources(raw: str | None, *, line_number: int) -> tuple[str, ...]:
    """把期望来源拆成分号分隔的文件名。为空是允许的（refuse 类通常没有）。"""
    text = _clean(raw)
    if text is None:
        return ()
    return tuple(item.strip() for item in text.split(_SOURCE_SEPARATOR) if item.strip())


def _build_case(raw: dict[str, object], *, line_number: int) -> EvalCase:
    """把一行原始数据转成 EvalCase。"""

    def cell(column: str) -> str | None:
        value = raw.get(column)
        return value if isinstance(value, str) else None

    question = _clean(cell("question"))
    if question is None:
        raise EvalCaseError("第 %d 行 question 为空" % line_number)

    behavior = _clean(cell("expected_behavior"))
    if behavior is None:
        raise EvalCaseError("第 %d 行 expected_behavior 为空" % line_number)
    if behavior not in ALLOWED_BEHAVIORS:
        raise EvalCaseError(
            "第 %d 行 expected_behavior：取值 %r 不在允许范围内，可选：%s"
            % (line_number, behavior, "、".join(sorted(ALLOWED_BEHAVIORS)))
        )

    category = _clean(cell("category"))
    if category is None:
        raise EvalCaseError("第 %d 行 category 为空" % line_number)

    return EvalCase(
        line_number=line_number,
        question=question,
        expected_behavior=behavior,
        category=category,
        expected_sources=_parse_sources(
            cell("expected_sources"), line_number=line_number
        ),
        notes=_clean(cell("notes")),
    )


def load_eval_cases(path: Path | None = None) -> list[EvalCase]:
    """读取并校验评测集。

    任何校验失败都抛 EvalCaseError，消息里带行号。
    读取时用 utf-8-sig，与文件本身的编码一致——否则 BOM 会并进第一列的
    列名里，整张表都认不出 question。
    """
    target = path if path is not None else EVAL_CASES_PATH
    if not target.is_file():
        raise EvalCaseError("评测集文件不存在：%s" % target)

    cases: list[EvalCase] = []
    with target.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header:
            raise EvalCaseError("评测集缺少表头：%s" % target)

        missing = [name for name in COLUMNS if name not in header]
        if missing:
            raise EvalCaseError(
                "评测集表头缺少列：%s。模板见 data/eval_cases.csv"
                % "、".join(missing)
            )

        # 表头占第 1 行，因此数据行从第 2 行开始计数
        for line_number, raw in enumerate(reader, start=2):
            cases.append(_build_case(raw, line_number=line_number))

    return cases


def summarize(cases: list[EvalCase]) -> EvalSummary:
    """统计两类用例各有多少条。"""
    return EvalSummary(
        answer_count=sum(1 for case in cases if case.should_answer),
        refuse_count=sum(1 for case in cases if not case.should_answer),
    )
