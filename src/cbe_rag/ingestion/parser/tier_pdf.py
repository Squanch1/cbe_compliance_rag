"""PDF 解析的层级实现。

两层：先用 PyMuPDF 提取文本层，不行再用 pdfplumber 做带表格识别的提取。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from cbe_rag.ingestion.parser.pdf_parser import (
    PDF_PARSER_VERSION,
    TextLine,
    build_blocks,
    parse_pdf,
)
from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ParsedDocument,
    SourceFormat,
)
from cbe_rag.ingestion.parser.text import normalise_whitespace
from cbe_rag.ingestion.parser.tier import ParseRequest

# 同一个 y 上的词视为同一行。取 1pt 容差——实测同一行的词 y 差在 0.5pt 内。
_SAME_LINE_TOLERANCE = 1.0

# 表格单元格之间的分隔符。用竖线是为了让大模型能看出列边界。
_CELL_SEPARATOR = " | "


class TextLayerPdfTier:
    """L1：PyMuPDF 提取文本层。

    对**有文本层的 PDF** 效果好且快（105 页约 0.3 秒）。
    扫描件会提取出极少的文本，由质量评估拦下后换下一层。
    """

    name = "pdf.text_layer"

    def parse(self, request: ParseRequest) -> ParsedDocument:
        return parse_pdf(request.raw_path, request.doc_id, request.title)


class TableAwarePdfTier:
    """L2：pdfplumber 提取，表格识别为独立的块。

    与 L1 的差别在于**表格处理**：L1 把表格文字当成散落的文本行，
    行列关系丢失；这一层把表格还原成 `TABLE` 块，单元格之间用竖线分隔，
    大模型能看出列边界。

    **已知局限**：

    1. 表格块按页追加在该页文本块之后，**页内顺序可能与原文不符**。
       精确合并需要把表格与文本行按 y 坐标统一排序，而 `build_blocks`
       只返回块、不返回块的行位置，暂时做不到。跨页顺序是正确的。
    2. **当前语料中没有表格**，这一层的主路径未经验证。等收到带表格的
       文档（如税率对照表）后需要重新验证。
    3. pdfplumber 比 PyMuPDF 慢，因此排在 L1 之后而非之前。
    """

    name = "pdf.table_aware"

    def parse(self, request: ParseRequest) -> ParsedDocument:
        import pdfplumber

        entries: list[Block] = []
        with pdfplumber.open(str(request.raw_path)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                entries.extend(self._page_blocks(page, page_number))

        blocks = [
            Block(
                type=entry.type,
                text=entry.text,
                order=index,
                level=entry.level,
                page=entry.page,
            )
            for index, entry in enumerate(entries)
        ]
        if not blocks:
            raise ValueError("pdfplumber 未提取到任何内容块")

        return ParsedDocument(
            doc_id=request.doc_id,
            title=request.title,
            source_format=SourceFormat.PDF,
            source_path=request.raw_path,
            parser_version=PDF_PARSER_VERSION,
            parsed_at=datetime.now(),
            blocks=blocks,
        )

    @staticmethod
    def _page_blocks(page: Any, page_number: int) -> list[Block]:
        """提取一页的内容块：先表格外的文字，再表格。"""
        tables = page.find_tables()
        table_boxes = [table.bbox for table in tables]

        words = [
            word
            for word in page.extract_words(extra_attrs=["size"])
            if not _inside_any(word, table_boxes)
        ]
        text_blocks = build_blocks(_words_to_lines(words, page_number))

        table_blocks = [_table_to_block(table, page_number) for table in tables]
        return list(text_blocks) + [b for b in table_blocks if b is not None]


def _inside_any(word: dict[str, Any], boxes: list[tuple[float, ...]]) -> bool:
    """判断一个词是否落在任一表格框内。"""
    center_y = (word["top"] + word["bottom"]) / 2
    center_x = (word["x0"] + word["x1"]) / 2
    for x0, top, x1, bottom in boxes:
        if x0 <= center_x <= x1 and top <= center_y <= bottom:
            return True
    return False


def _words_to_lines(words: list[dict[str, Any]], page_number: int) -> list[TextLine]:
    """把词按 y 聚成行。

    复用 pdf_parser 的 TextLine 结构，这样段落合并的规则
    （行距阈值、跨页断开）与本层保持一致，不必重写一遍。
    """
    if not words:
        return []

    ordered = sorted(words, key=lambda word: (word["top"], word["x0"]))
    rows: list[list[dict[str, Any]]] = []
    for word in ordered:
        if rows and abs(word["top"] - rows[-1][0]["top"]) <= _SAME_LINE_TOLERANCE:
            rows[-1].append(word)
        else:
            rows.append([word])

    lines: list[TextLine] = []
    for row in rows:
        text = normalise_whitespace(" ".join(word["text"] for word in row))
        if not text:
            continue
        lines.append(
            TextLine(
                page=page_number,
                text=text,
                size=max(float(word.get("size") or 0.0) for word in row),
                top=min(word["top"] for word in row),
            )
        )
    return lines


def _table_to_block(table: Any, page_number: int) -> Block | None:
    """把 pdfplumber 的表格转成 TABLE 块。

    单元格之间用竖线分隔、行之间用换行——让大模型能看出行列结构。
    空行与全空单元格会被跳过，否则表格块的噪声比信息还多。
    """
    rows = table.extract() or []
    lines: list[str] = []
    for row in rows:
        cells = [normalise_whitespace(str(cell)) if cell else "" for cell in row]
        if not any(cells):
            continue
        lines.append(_CELL_SEPARATOR.join(cells))
    if not lines:
        return None

    return Block(
        type=BlockType.TABLE,
        text="\n".join(lines),
        order=0,
        page=page_number,
    )
