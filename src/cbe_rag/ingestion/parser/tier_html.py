"""HTML 解析的层级实现。

两层：先按站点选择器精确提取，不行再退回通用正文提取。
"""

from __future__ import annotations

import re
from datetime import datetime

from bs4 import BeautifulSoup

from cbe_rag.ingestion.parser.html_parser import (
    HTML_PARSER_VERSION,
    HtmlParseError,
    parse_html,
)
from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ParsedDocument,
    SourceFormat,
)
from cbe_rag.ingestion.parser.text import normalise_whitespace
from cbe_rag.ingestion.parser.tier import ParseRequest

class SelectorHtmlTier:
    """L1：按站点配置的选择器精确提取正文。

    精确，但依赖配置——未配置的站点会直接报错，由路由换下一层。
    这是**有意为之**：通用规则在真实页面上会残留导航与页脚
    （实测某帮助页残留三个噪声块），宁可报错也不产脏数据。
    """

    name = "html.selector"

    def parse(self, request: ParseRequest) -> ParsedDocument:
        if not request.source_url:
            raise HtmlParseError(
                "缺少 source_url，无法确定站点选择器。"
                "这一层依赖站点配置，元数据不全时应由下一层接手"
            )
        return parse_html(request.raw_path, request.doc_id, request.source_url)


class GenericHtmlTier:
    """L2：不依赖站点配置的通用正文提取。

    用 trafilatura 自动识别正文区域。它对主流站点做过调优，但**不保证
    剔除干净**——导航与相关链接可能混入。因此它排在 L1 之后：
    L1 能配就配，配不了才用它兜底。
    """

    name = "html.generic"

    def parse(self, request: ParseRequest) -> ParsedDocument:
        import trafilatura

        html = request.raw_path.read_text(encoding="utf-8", errors="replace")
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if not extracted:
            raise HtmlParseError("通用提取未取得正文，可能是纯脚本页面")

        blocks = _blocks_from_text(extracted)
        if not blocks:
            raise HtmlParseError("通用提取出的正文为空")

        return ParsedDocument(
            doc_id=request.doc_id,
            title=request.title,
            source_format=SourceFormat.HTML,
            source_path=request.raw_path,
            parser_version=HTML_PARSER_VERSION,
            parsed_at=datetime.now(),
            blocks=blocks,
        )


def _blocks_from_text(text: str) -> list[Block]:
    """把纯文本切成块。

    **按行切，不是按空行切。** trafilatura 输出的段落之间用换行分隔，
    不是空行——实测按空行切会把整篇正文并成一个块。兼容处理：
    空行直接跳过，因此两种分隔方式都能切对。

    **标题无从判断**：纯文本没有标签，且 trafilatura 会按「正文密度」
    过滤，短标题行常被当成装饰丢掉。实测某帮助页的七个问题标题全部
    丢失。因此这一层的 `headings` 恒为 0，是它作为兜底方案的固有代价。
    """
    blocks: list[Block] = []
    for chunk in text.splitlines():
        cleaned = normalise_whitespace(chunk)
        if not cleaned:
            continue
        blocks.append(
            Block(
                type=BlockType.PARAGRAPH,
                text=cleaned,
                order=len(blocks),
            )
        )
    return blocks
