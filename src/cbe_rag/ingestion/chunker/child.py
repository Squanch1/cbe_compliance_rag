"""子块切分。

子块是建立向量索引与执行检索的单位，目标约 300 token。

**关键不变式：子块必须是父块的连续子串。** 界面靠这条不变式在父块
正文里高亮命中的子块（见 docs/spec/02-architecture.md 第 6.5 节）。
因此这里返回的不只是文本，还有子块在父块文本中的字符区间——
只给文本的话，界面得自己用字符串查找定位，而同一段文字在一个父块里
可能重复出现，查找会命中错误的那一处。
"""

from __future__ import annotations

from dataclasses import dataclass

from cbe_rag.ingestion.parser.schema import Block

# 子块 token 目标。达到就切，因此实际大小会略微超过——
# 与父块同理，切分只能发生在块与块之间，块内没有更小的边界可用。
CHILD_TARGET_TOKENS = 300

# 块之间用换行分隔。块本身是段落，用换行保留段落结构，
# 比全部挤成一行更利于大模型阅读。
BLOCK_SEPARATOR = "\n"


class ChildSplitError(Exception):
    """切分无法继续时抛出。"""


@dataclass(frozen=True)
class ChildSpan:
    """子块在父块文本中的位置与内容。

    start / end: 在父块文本中的字符区间，左闭右开
    text:        该区间的文本，恒等于 parent_text[start:end]
    """

    start: int
    end: int
    text: str


@dataclass(frozen=True)
class ParentChunking:
    """一个父块的切分结果。"""

    text: str
    children: list[ChildSpan]


def chunk_parent(
    blocks: list[Block],
    counts: list[int],
    *,
    target: int = CHILD_TARGET_TOKENS,
) -> ParentChunking:
    """把一个父块切成子块。

    counts 是每个块对应的 token 数，由调用方预先算好——
    切分逻辑本身不该关心用哪个 tokenizer。
    """
    if len(blocks) != len(counts):
        raise ChildSplitError(
            "块与 token 计数数量不一致：%d 个块，%d 个计数"
            % (len(blocks), len(counts))
        )
    if not blocks:
        return ParentChunking(text="", children=[])

    # 拼父块文本，同时记下每个块的字符区间——
    # 区间在拼接过程中顺手算，比事后用字符串查找可靠
    pieces: list[str] = []
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for index, block in enumerate(blocks):
        if index > 0:
            pieces.append(BLOCK_SEPARATOR)
            cursor += len(BLOCK_SEPARATOR)
        start = cursor
        cursor += len(block.text)
        bounds.append((start, cursor))
        pieces.append(block.text)
    parent_text = "".join(pieces)

    children: list[ChildSpan] = []
    span_start: int | None = None
    span_end = 0
    accumulated = 0

    for index, count in enumerate(counts):
        if span_start is None:
            span_start = bounds[index][0]
        span_end = bounds[index][1]
        accumulated += count

        if accumulated >= target:
            children.append(
                ChildSpan(
                    start=span_start,
                    end=span_end,
                    text=parent_text[span_start:span_end],
                )
            )
            span_start, accumulated = None, 0

    # 不足目标的余量也要产出，否则尾部内容会丢
    if span_start is not None:
        children.append(
            ChildSpan(
                start=span_start,
                end=span_end,
                text=parent_text[span_start:span_end],
            )
        )

    return ParentChunking(text=parent_text, children=children)
