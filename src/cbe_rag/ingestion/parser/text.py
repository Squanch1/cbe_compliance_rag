"""解析器共用的文本处理。

HTML 与 PDF 的提取机制不同，但清洗规则一致，因此放在这里共用。
"""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalise_whitespace(text: str) -> str:
    """把连续空白折叠成一个空格，并去掉首尾空白。

    源码里的换行与缩进对正文没有意义：HTML 的排版缩进、PDF 的
    行内换行都属于版式噪声，不是内容的一部分。
    """
    return _WHITESPACE.sub(" ", text).strip()
