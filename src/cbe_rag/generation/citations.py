"""引用解析与校验。

回答里的 `[n]` 是唯一能证明「这句话有出处」的东西，所以它得由程序核对，
而不是相信模型自己说「根据以上材料」。

**这里能做什么、不能做什么，要说清楚：**

- 能拦住引用了不存在编号的回答。编号越界是模型幻觉的常见形式，而且
  不拦住的话，一个 `[7]` 会被原样呈现给使用者，看着像有出处。
- 能发现一条引用都没有的回答，交给调用方降级为「依据不足」。
- **不能**判断引用是否恰当。模型完全可能把 `[2]` 挂在只由 `[1]` 支持的
  结论上，程序看不出这个差别。这一层的可靠性靠评测集衡量，不靠这里兜。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cbe_rag.retrieval.models import RetrievedParent

# 引用标记：方括号包一个数字。
#
# 全角括号一并接受——中文回答里模型偶尔会写成 ［1］，只认半角的话那种
# 引用会被当成「没有引用」，而实际有，判反了比不判更糟。
#
# 暂不支持 [1,2] 这种逗号合并写法：提示词里明确要求的是 [1][2]。
# 实测如果模型常用那种写法，再放宽。
_CITATION = re.compile(r"[\[［](\d+)[\]］]")


@dataclass(frozen=True)
class CitationReport:
    """一次引用校验的结论。"""

    # 被引用到的材料，按材料原本的顺序，不按引用出现的顺序
    cited: list[RetrievedParent]
    # 回答里出现、但超出材料范围的编号
    invalid_numbers: list[int]

    @property
    def has_any(self) -> bool:
        """有没有至少一处合法引用。没有就该降级为「依据不足」。"""
        return bool(self.cited)


def parse_citations(answer: str) -> list[int]:
    """按出现顺序取出回答里的引用编号，重复的只留一次。"""
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if number in seen:
            continue
        seen.add(number)
        numbers.append(number)
    return numbers


def check_citations(
    answer: str, parents: list[RetrievedParent]
) -> CitationReport:
    """核对回答里的引用编号。

    编号从 1 开始，与提示词里的编号一致；`[0]` 与超出材料数的都算越界。
    """
    cited: list[RetrievedParent] = []
    invalid: list[int] = []

    for number in parse_citations(answer):
        if not 1 <= number <= len(parents):
            invalid.append(number)
            continue
        cited.append(parents[number - 1])

    # 材料顺序比引用顺序更稳定，输出时按前者排
    order = {parent.parent_id: index for index, parent in enumerate(parents)}
    cited.sort(key=lambda parent: order[parent.parent_id])
    return CitationReport(cited=cited, invalid_numbers=invalid)
