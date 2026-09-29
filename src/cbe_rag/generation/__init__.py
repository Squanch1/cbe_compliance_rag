"""生成：把检索到的材料组装成提示词，调用大模型，并校验引用。

所有回答必须携带可追溯的引用；给不出引用时显式声明依据不足，不编造。
"""

from cbe_rag.generation.answer import (
    NO_EVIDENCE_TEXT,
    Answer,
    generate_answer,
)
from cbe_rag.generation.citations import (
    CitationReport,
    check_citations,
    parse_citations,
)
from cbe_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
    has_unmarked_effective_date,
)
from cbe_rag.generation.service import (
    REFUSAL_TEXT,
    QaResult,
    answer_question,
)

__all__ = [
    "NO_EVIDENCE_TEXT",
    "REFUSAL_TEXT",
    "SYSTEM_PROMPT",
    "Answer",
    "CitationReport",
    "QaResult",
    "answer_question",
    "build_context",
    "build_user_prompt",
    "check_citations",
    "generate_answer",
    "has_unmarked_effective_date",
    "parse_citations",
]
