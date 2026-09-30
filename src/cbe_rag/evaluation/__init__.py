"""评测：读评测集、跑用例、标定阈值。

评测集是手工维护的 CSV，程序读它、跑它、报告结果。指标与基线写入
docs/spec/05-acceptance.md。
"""

from cbe_rag.evaluation.cases import (
    ALLOWED_BEHAVIORS,
    EXPECTED_ANSWER,
    EXPECTED_REFUSE,
    EvalCase,
    EvalCaseError,
    load_eval_cases,
    summarize,
)

__all__ = [
    "ALLOWED_BEHAVIORS",
    "EXPECTED_ANSWER",
    "EXPECTED_REFUSE",
    "EvalCase",
    "EvalCaseError",
    "load_eval_cases",
    "summarize",
]
