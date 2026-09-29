"""PDF 解析器。

把 PDF 转成统一中间表示。

与 HTML 解析器的关键差异：PDF 没有 `<h1>` 这类层级标记，标题只能靠
字号推断；也没有 `<p>` 标出段落边界，只能靠行距与坐标。因此这里
分两步——先把 PDF 提取成干净的行，再判断层级、组装成块。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from cbe_rag.ingestion.parser.text import normalise_whitespace

# 页码的文本模式，形如 "5/105"
_PAGE_NUMBER_PATTERN = re.compile(r"^\d+\s*/\s*\d+$")

# 页面高度这个比例以下算页脚区域。
# 实测欧盟文档的页码在 y≈783-794（页高 842），约 93% 处。
_BOTTOM_ZONE_RATIO = 0.9


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
