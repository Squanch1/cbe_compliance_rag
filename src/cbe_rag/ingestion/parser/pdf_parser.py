"""PDF 解析器。

把 PDF 转成统一中间表示。

与 HTML 解析器的关键差异：PDF 没有 `<h1>` 这类层级标记，标题只能靠
字号推断；也没有 `<p>` 标出段落边界，只能靠行距与坐标。因此这里
分两步——先把 PDF 提取成干净的行，再判断层级、组装成块。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ParsedDocument,
    SourceFormat,
)
from cbe_rag.ingestion.parser.text import normalise_whitespace

# 解析器版本。与 HTML 解析器的版本各自独立演进。
PDF_PARSER_VERSION = "0.1.0"

# 页码的文本模式，形如 "5/105"
_PAGE_NUMBER_PATTERN = re.compile(r"^\d+\s*/\s*\d+$")

# 页面高度这个比例以下算页脚区域。
# 实测欧盟文档的页码在 y≈783-794（页高 842），约 93% 处。
_BOTTOM_ZONE_RATIO = 0.9

# 段落边界的行距阈值。实测欧盟文档：正常行距约 13.8pt
# （波动范围 11.5-14.5），段落间距约 25.8pt（波动范围 24.6-26.4），
# 两档之间有明显空档，取 20pt 作分界很安全。
_PARAGRAPH_GAP = 20.0

# y 坐标差小于此值视为同一行上的并列块，
# 例如目录里的章节号与其后的标题。
_SAME_LINE_TOLERANCE = 1.0


class PdfParseError(Exception):
    """PDF 解析无法继续时抛出。"""


@dataclass(frozen=True)
class TextLine:
    """PDF 里的一行文本。

    page: 页码，从 1 开始
    text: 该行的正文
    size: 该行最大的字号，用于推断标题层级
    top:  该行上边缘的 y 坐标，用于判断段落边界
    """

    page: int
    text: str
    size: float
    top: float


def _join_spans(spans: list[Any]) -> str:
    """拼接一行里的 span 文本。

    **直接拼接，不加分隔符。** PyMuPDF 的 span 文本自带必要的空格
    （small caps 被拆成的 'MPORT ' 就带尾随空格），额外插入分隔符
    反而会切断单词——实测把 "Article" 变成了 "Arti cle"，
    也会破坏连续大写（IMPORT 变成 I MPORT）。
    """
    return "".join(span["text"] for span in spans)


def _is_page_number(text: str, top: float, page_height: float) -> bool:
    """判断某一行是否页脚页码。

    位置与文本模式**必须同时满足**：正文可以延伸到页面底部
    （实测欧盟文档第 41 页就有），而正文里出现 "1/2" 这种分数
    也很正常。只看其中一个条件都会误删正文。
    """
    if top < page_height * _BOTTOM_ZONE_RATIO:
        return False
    return bool(_PAGE_NUMBER_PATTERN.match(text))


def _line_size(spans: list[Any]) -> float:
    """取一行中最大的字号，作为该行的代表字号。"""
    sizes = [round(span["size"], 1) for span in spans if span["text"].strip()]
    return max(sizes) if sizes else 0.0


def extract_lines(pdf_path: Path) -> list[TextLine]:
    """逐页提取文本行，并剔除页码。

    只负责提取与清洗，不判断标题层级——那是下一步的事。
    """
    # 先判绝对路径再判存在性：相对路径先报「文件不存在」会把
    # 排查方向带偏（见 CLAUDE.md 5.2）
    if not pdf_path.is_absolute():
        raise PdfParseError("pdf_path 必须是绝对路径，收到：%s" % pdf_path)
    if not pdf_path.is_file():
        raise PdfParseError("PDF 文件不存在：%s" % pdf_path)

    try:
        document = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PdfParseError(
            "无法打开 PDF：%s（%s: %s）" % (pdf_path, type(exc).__name__, exc)
        ) from exc

    lines: list[TextLine] = []
    try:
        for page_index, page in enumerate(document):
            page_height = page.rect.height
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:  # 非文本块（图片等）
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    text = normalise_whitespace(_join_spans(spans))
                    if not text:
                        continue
                    top = float(line["bbox"][1])
                    if _is_page_number(text, top, page_height):
                        continue
                    lines.append(
                        TextLine(
                            page=page_index + 1,
                            text=text,
                            size=_line_size(spans),
                            top=top,
                        )
                    )
    finally:
        document.close()

    return lines


def body_font_size(lines: list[TextLine]) -> float:
    """统计正文字号。

    按**行数**取众数，而不是按字符数。两者在正常排版下结论一致，
    但行数更直观：正文占的行最多，这一点一眼能解释清楚。
    """
    if not lines:
        raise PdfParseError("没有可供统计的行，无法确定正文字号")
    counts = Counter(round(line.size, 1) for line in lines)
    top_count = max(counts.values())
    # 平局时取较小的字号：标题总是比正文大，较小的那个更可能是正文。
    # 不能用 most_common，它在平局时按出现顺序任选，
    # 结果取决于文档里哪一行排在前——同一份文档换个排版就变了。
    return min(size for size, count in counts.items() if count == top_count)


def heading_levels(
    lines: list[TextLine], *, body_size: float | None = None
) -> dict[float, int]:
    """把大于正文字号的字号映射成标题层级。

    字号越大层级越浅：最大的是 1 级，次大的是 2 级，依此类推。
    层级由字号**相对关系**决定而不是写死数值，换一份排版不同的
    文档不用改代码。

    小于或等于正文字号的都不算标题——实测欧盟文档里 11pt 是
    次级正文（免责声明、前言），不是标题。

    `body_size` 传了就用它，不传才从 lines 里算。**分片段调用时必须传**：
    正文字号是整份文档的属性，按片段各算一次会因片段的字号分布不同
    而得出不同结论。实测踩过——某解析器按页调用，单页里 12pt 不是众数，
    于是 12pt 反被当成标题，72% 的块被误判。
    """
    if not lines:
        return {}

    body = round(body_size if body_size is not None else body_font_size(lines), 1)
    # **先统一取整再比较，两步不能用不同的值。** 字号可能带浮点误差
    # （pdfplumber 实测给出过 12.000000000000028），若用原值比较、
    # 用取整后的值入集合，正文自己就会混进标题集合，整页正文都被判成
    # 标题——实测某份 105 页 PDF 因此产出 1333 个假标题。
    sizes = {round(line.size, 1) for line in lines}
    larger = sorted((size for size in sizes if size > body), reverse=True)
    return {size: index + 1 for index, size in enumerate(larger)}


def _continues(previous: TextLine, current: TextLine) -> bool:
    """判断当前行是否延续上一行所在的段落。"""
    if current.page != previous.page:
        # 跨页一律断开。判断跨页续段需要额外信息，断错的代价是一段
        # 被切成两半，而误合的代价是把两页的无关内容并成一段，
        # 后者对检索的干扰更大。
        return False

    gap = current.top - previous.top
    if gap < _SAME_LINE_TOLERANCE:
        # 同一行上的并列块，例如目录里的章节号与标题
        return True
    return gap <= _PARAGRAPH_GAP


def build_blocks(
    lines: list[TextLine], *, body_size: float | None = None
) -> list[Block]:
    """把行合并成块。

    字号大于正文的行单独成标题块；其余行按行距合并成段落：
    同一页内行距正常就续段，行距明显变大或跨页就另起一段。

    PDF 里每行都是独立对象，段落边界只能靠行距推断——这是与 HTML
    最大的不同，HTML 有 `<p>` 直接标出边界。

    `body_size` 透传给 heading_levels，分片段调用时必须传，
    理由见那里。
    """
    if not lines:
        return []

    levels = heading_levels(lines, body_size=body_size)
    blocks: list[Block] = []
    pending: list[str] = []
    previous: TextLine | None = None

    def flush() -> None:
        if not pending:
            return
        blocks.append(
            Block(
                type=BlockType.PARAGRAPH,
                text=normalise_whitespace(" ".join(pending)),
                order=len(blocks),
            )
        )
        pending.clear()

    for line in lines:
        size = round(line.size, 1)
        if size in levels:
            flush()
            blocks.append(
                Block(
                    type=BlockType.HEADING,
                    text=line.text,
                    order=len(blocks),
                    level=levels[size],
                )
            )
        elif previous is not None and _continues(previous, line):
            pending.append(line.text)
        else:
            flush()
            pending.append(line.text)
        previous = line

    flush()
    return blocks


def parse_pdf(
    pdf_path: Path,
    doc_id: str,
    title: str,
    *,
    parsed_at: datetime | None = None,
) -> ParsedDocument:
    """把 PDF 文件解析成 ParsedDocument。

    **标题由调用方传入，不从 PDF 里猜。** PDF 没有 `<h1>` 等价物，
    封面标题常常跨多个字号与行——实测欧盟文档的标题被拆成 22pt 的
    「Explanatory Notes」与 18pt 的「VAT e-commerce rules」两部分，
    靠字号猜出来的往往不完整。清单里已有人工登记的标题，质量更高。

    parsed_at 可注入，便于测试固定时间戳。
    """
    # 先判绝对路径再判存在性：相对路径先报「文件不存在」会把
    # 排查方向带偏（见 CLAUDE.md 5.2）
    if not pdf_path.is_absolute():
        raise PdfParseError("pdf_path 必须是绝对路径，收到：%s" % pdf_path)
    if not pdf_path.is_file():
        raise PdfParseError("PDF 文件不存在：%s" % pdf_path)

    lines = extract_lines(pdf_path)
    if not lines:
        # 扫描件这类没有文本层的 PDF 会走到这里
        raise PdfParseError("PDF 里没有提取到任何文本：%s" % pdf_path)

    blocks = build_blocks(lines)
    if not blocks:
        raise PdfParseError("PDF 未解析出任何内容块：%s" % pdf_path)

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_format=SourceFormat.PDF,
        source_path=pdf_path,
        parser_version=PDF_PARSER_VERSION,
        parsed_at=parsed_at if parsed_at is not None else datetime.now(),
        blocks=blocks,
    )
