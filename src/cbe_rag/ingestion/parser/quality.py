"""解析结果的质量评估。

每层解析完先评估，合格才采用，不合格就换下一层。评估不是「尽力而为」
的参考分，而是**决定流程走向的门禁**，因此失败时必须说清是哪条指标
不达标。

指标都从 ParsedDocument 直接算得，不依赖原始文件——
这样无论哪一层产出的结果，都用同一把尺子衡量。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from cbe_rag.ingestion.parser.schema import Block, BlockType, ParsedDocument

# 乱码字符用**黑名单**判定，不用白名单。
#
# 乱码有明确特征：编码失败留下的替换符、控制字符、字体映射错误落入的
# 私用区。而合法字符千变万化——欧元符号、破折号、弯引号、书名号都是
# 正常文本，白名单永远列不全。实测踩过：白名单把欧元符号和破折号
# 都判成了乱码。
_GARBAGE_CHAR = re.compile(
    "["
    "\ufffd"                      # \u66ff\u6362\u5b57\u7b26\uff1a\u7f16\u7801\u5931\u8d25\u7684\u6807\u5fd7
    "\x00-\x08\x0b\x0c\x0e-\x1f"  # \u63a7\u5236\u5b57\u7b26\uff08\u5236\u8868\u7b26\u4e0e\u6362\u884c\u9664\u5916\uff09
    "\ue000-\uf8ff"               # \u79c1\u7528\u533a\uff1a\u5b57\u4f53\u6620\u5c04\u9519\u8bef\u5e38\u843d\u5728\u8fd9\u91cc
    "]"
)

# 短于此长度的段落类块算「碎片」。标题短是正常的，因此不计入。
_FRAGMENT_CHARS = 10

# 参与碎片率统计的块类型：正文类，不含标题
_BODY_TYPES = frozenset({BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.TABLE})


@dataclass(frozen=True)
class QualityThresholds:
    """判定的阈值。

    取值来自三篇真实语料的实测分布：

        指标      实测范围            阈值    余量
        字符数    1020 - 268787       200     只拦「几乎没解析出东西」
        乱码率    0.00% - 0.11%       1%      约 10 倍
        碎片率    0.0%  - 0.7%        20%     约 30 倍
        重复率    0.0%  - 8.7%        25%     约 3 倍

    阈值的作用是拦住**明显坏掉的解析**（扫描件、字体编码错误、排版识别
    失败），不是优中选优。因此留足余量，宁可漏放也不误杀——误杀会让
    好文档白白走一遍人工流程，代价比漏放高得多。

    **重复率这条有已知局限**：它区分不了「页眉页脚残留」与「文档自身的
    结构重复」。实测那份 8.7% 全部来自 Example 章节的重复小标题，是
    文档固有特征而非缺陷。当前两个解析器都已剔除页眉页脚，因此这个
    指标实际起的是**回归监控**作用——只有解析器被改坏了它才会飙升。
    """

    # 不设块数下限：ParsedDocument 已强制 blocks 至少一个，
    # 字符数那一项也在把关，再加一个永不触发的检查只会让它看起来
    # 像一道防护，实际是死代码。
    min_total_chars: int = 200
    max_garbage_ratio: float = 0.01
    max_fragment_ratio: float = 0.20
    max_duplicate_ratio: float = 0.25


@dataclass(frozen=True)
class QualityReport:
    """一份解析结果的质量评分。

    metrics 无论通过与否都完整保留：调阈值时需要看分布，
    而分布只能从这些原始分数里来。
    """

    passed: bool
    metrics: dict[str, float]
    failures: list[str]


def _garbage_ratio(text: str) -> float:
    """乱码字符占比。

    用「一次性剔除允许字符，看剩多少」而不是逐字符判断：
    同样是一遍扫描，但走 C 层而不是 Python 循环，几十万字符的文档
    差出两个数量级的耗时。
    """
    if not text:
        # 空文本按全乱码处理。它本来也会在字符数那一项失败，
        # 这里返回 1.0 是为了让只查乱码率的调用方也不会放行。
        return 1.0
    return len(_GARBAGE_CHAR.findall(text)) / len(text)


def _fragment_ratio(blocks: list[Block]) -> float:
    """正文类块中的碎片占比。

    只统计正文类块：标题本来就可能很短（如 "1.1"），
    把它算作碎片会误判。
    """
    body = [block for block in blocks if block.type in _BODY_TYPES]
    if not body:
        return 0.0
    fragments = sum(1 for block in body if len(block.text) < _FRAGMENT_CHARS)
    return fragments / len(body)


def _duplicate_ratio(blocks: list[Block]) -> float:
    """重复块占比。

    页眉页脚没剔干净时，同样的文字会在每页出现一次，
    这个指标就是用来发现那种情况的。
    """
    if not blocks:
        return 0.0
    counts = Counter(block.text for block in blocks)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(blocks)


def assess(
    document: ParsedDocument,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """评估一份解析结果是否合格。

    不合格时 failures 逐条写清**哪个指标不达标、实际值是多少**。
    只说「质量不合格」既没法定位问题，也没法调阈值——而阈值正是
    接下来要拿这些分数去标定的东西。
    """
    limits = thresholds if thresholds is not None else QualityThresholds()
    text = "\n".join(block.text for block in document.blocks)

    metrics = {
        "total_chars": float(len(text)),
        "blocks": float(len(document.blocks)),
        "headings": float(
            sum(1 for block in document.blocks if block.type is BlockType.HEADING)
        ),
        "garbage_ratio": _garbage_ratio(text),
        "fragment_ratio": _fragment_ratio(document.blocks),
        "duplicate_ratio": _duplicate_ratio(document.blocks),
    }

    failures: list[str] = []
    if metrics["total_chars"] < limits.min_total_chars:
        failures.append(
            "总字符数 %.0f 低于下限 %d"
            % (metrics["total_chars"], limits.min_total_chars)
        )
    if metrics["garbage_ratio"] > limits.max_garbage_ratio:
        failures.append(
            "乱码率 %.1f%% 超过上限 %.1f%%"
            % (metrics["garbage_ratio"] * 100, limits.max_garbage_ratio * 100)
        )
    if metrics["fragment_ratio"] > limits.max_fragment_ratio:
        failures.append(
            "碎片块占比 %.0f%% 超过上限 %.0f%%"
            % (metrics["fragment_ratio"] * 100, limits.max_fragment_ratio * 100)
        )
    if metrics["duplicate_ratio"] > limits.max_duplicate_ratio:
        failures.append(
            "重复块占比 %.0f%% 超过上限 %.0f%%"
            % (metrics["duplicate_ratio"] * 100, limits.max_duplicate_ratio * 100)
        )

    return QualityReport(passed=not failures, metrics=metrics, failures=failures)
