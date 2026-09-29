"""父块切分。

父块是喂给大模型的上下文单位，目标约 1500 token。

**不能简单按标题切。** 实测欧盟文档：按标题切出 73 组，token 中位数
只有 70，而最大的一组有 13007——分布极度不均，跨了三个数量级。
原因是文档里大量是目录条目与小标题，每个只带一两行内容。
若按标题切，三分之二的父块会小到与子块无异，父子分块完全退化。

因此改为按 token 数切分，标题作为**优先**断点：累积够了下限之后
遇到标题就断，到了上限则强制断。
"""

from __future__ import annotations

from cbe_rag.ingestion.parser.schema import Block, BlockType

# 父块 token 的上下限。目标值 1500 落在两者中间，允许 ±33% 浮动，
# 是为了让断点尽量落在标题上——严格卡 1500 会把句子拦腰截断。
PARENT_LOWER_TOKENS = 1000
PARENT_UPPER_TOKENS = 2000


class ParentSplitError(Exception):
    """切分无法继续时抛出。"""


def split_parents(
    blocks: list[Block],
    counts: list[int],
    *,
    lower: int = PARENT_LOWER_TOKENS,
    upper: int = PARENT_UPPER_TOKENS,
) -> list[list[Block]]:
    """把块序列切成若干父块。

    counts 是每个块对应的 token 数，由调用方预先算好——
    切分逻辑本身不该关心用哪个 tokenizer。

    **上限是软上限**：切分只能发生在块与块之间，因此实际大小会略微
    超过上限，超出量不超过单个最大块（实测欧盟文档超出约 240 token）。
    改成「加块之前判断」能让上限真正生效，但会在末尾留下百来 token 的
    孤儿块——那比超出 200 token 更糟，父块大小不均对检索和生成都不利。

    单个块超过上限时切不开（块内没有更小的边界可用），
    它会独占一个父块。PDF 里超长段落可能触发这种情况。
    """
    if len(blocks) != len(counts):
        raise ParentSplitError(
            "块与 token 计数数量不一致：%d 个块，%d 个计数"
            % (len(blocks), len(counts))
        )
    if lower > upper:
        raise ParentSplitError("下限（%d）不能大于上限（%d）" % (lower, upper))
    if not blocks:
        return []

    parents: list[list[Block]] = []
    current: list[Block] = []
    accumulated = 0

    for block, count in zip(blocks, counts):
        # 累积够了下限、又正好遇到标题：在这里断，断点落在语义边界上
        if current and accumulated >= lower and block.type is BlockType.HEADING:
            parents.append(current)
            current, accumulated = [], 0

        current.append(block)
        accumulated += count

        # 到了上限还没遇到标题也要断，否则超长章节的父块会越过模型上限
        # （实测最大的章节有 13007 token，超过 bge-m3 的 8192）
        if accumulated >= upper:
            parents.append(current)
            current, accumulated = [], 0

    if current:
        parents.append(current)
    return parents
