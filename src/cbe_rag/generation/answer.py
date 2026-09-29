"""一次回答的生成。

组装提示词、调用模型、校验引用。校验不通过的回答要降级，不能原样返回
——一段没有出处的结论，使用者看不出来它没有依据。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cbe_rag.generation.citations import CitationReport, check_citations
from cbe_rag.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    has_unmarked_effective_date,
)
from cbe_rag.retrieval.models import RetrievedParent
from cbe_rag.storage.bailian_client import BailianClient

# 降级时替换进去的文本。写清「为什么答不了」，而不是只说一句「无法回答」
# ——使用者会想知道是不是自己问错了。
NO_EVIDENCE_TEXT = (
    "依据不足，无法回答。检索到的材料里没有能支撑这个问题的内容。"
)


@dataclass(frozen=True)
class Answer:
    """一次生成的完整结果。

    **text 是最终要展示给使用者的文本。** 降级时它已被替换成依据不足的
    声明，调用方直接展示即可，不必再判断 degraded——少一处判断，就少
    一个「忘了判断」的机会。

    raw_text 保留模型的原始输出，供排查：降级时最想知道的就是它到底写了
    什么。

    notes 是需要额外告知使用者的话（时效冲突、引用被剔除等），由调用方
    决定怎么呈现。
    """

    text: str
    raw_text: str
    citations: CitationReport
    degraded: bool
    notes: list[str] = field(default_factory=list)


def generate_answer(
    question: str,
    parents: list[RetrievedParent],
    client: BailianClient,
) -> Answer:
    """组装提示词、调用模型、校验引用，返回可以直接展示的回答。

    没有材料时**不调用模型**：调用只会得到一段没有依据的话，然后被校验
    拦下——白花一次钱，还多一次把幻觉写进日志的机会。
    """
    if not parents:
        return Answer(
            text=NO_EVIDENCE_TEXT,
            raw_text="",
            citations=CitationReport(cited=[], invalid_numbers=[]),
            degraded=True,
        )

    completion = client.complete(
        SYSTEM_PROMPT, build_user_prompt(question, parents)
    )
    report = check_citations(completion.text, parents)

    notes: list[str] = []
    if has_unmarked_effective_date(parents):
        # 02-architecture 6.2.3：多份规定并存而部分未标日期时，必须显式
        # 告知无法判断孰新，不得默认选一份。程序来兜这一条，不指望模型
        # 每次都记得说。
        notes.append(
            "检索到的材料里有未标注生效日期的，无法判断它与其他材料孰新。"
        )

    if not report.has_any:
        return Answer(
            text=NO_EVIDENCE_TEXT,
            raw_text=completion.text,
            citations=report,
            degraded=True,
            notes=notes,
        )

    if report.invalid_numbers:
        # 越界的编号已从引用里剔除。这件事要说出来：模型编了一个来源，
        # 意味着它对这一段的把握值得怀疑。
        notes.append(
            "模型引用了不存在的材料编号（%s），已从引用中剔除。"
            % "、".join(str(number) for number in report.invalid_numbers)
        )

    return Answer(
        text=completion.text,
        raw_text=completion.text,
        citations=report,
        degraded=False,
        notes=notes,
    )
