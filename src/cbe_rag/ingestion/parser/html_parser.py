"""HTML 解析器。

把网页转成统一中间表示的 Block 列表。

正文位置按**站点配置的选择器**定位，不做通用启发式判断。原因是实测过的：
某帮助页用「排除 header/footer/nav/aside」的通用规则，仍然残留了
语言选择器、相关文章、页脚三个噪声块。产出脏数据比直接报错更难排查。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ParsedDocument,
    SourceFormat,
)
from cbe_rag.ingestion.parser.text import normalise_whitespace

# 解析器版本。解析规则变化时递增，便于回溯「这份产物是哪一版解析器生成的」。
PARSER_VERSION = "0.1.0"

# 站点到正文容器的映射。
# 接入新来源时在这里加一条，加之前先打开真实页面确认容器选择器。
SITE_CONTENT_SELECTORS: dict[str, str] = {
    "sellercentral.amazon.com": "#help-content",
}

_HEADING_LEVELS: dict[str, int] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}

_EXTRACT_TAGS: tuple[str, ...] = tuple(_HEADING_LEVELS) + ("p", "li", "table")
_REMOVED_TAGS: tuple[str, ...] = ("script", "style")


class HtmlParseError(Exception):
    """HTML 解析无法继续时抛出。"""


def content_selector_for(url: str) -> str:
    """按 URL 的域名查正文容器选择器。

    未配置的站点直接报错，要求先看真实 DOM 再接入，
    而不是退回一套猜出来的通用规则。
    """
    host = (urlparse(url).hostname or "").lower()
    if host.startswith("www."):
        host = host[len("www.") :]

    selector = SITE_CONTENT_SELECTORS.get(host)
    if selector is None:
        raise HtmlParseError(
            "站点未配置正文容器选择器：%s。已配置：%s。"
            "接入新来源前请打开真实页面确认容器，"
            "并加入 html_parser.SITE_CONTENT_SELECTORS。"
            % (host, "、".join(sorted(SITE_CONTENT_SELECTORS)))
        )
    return selector


def _block_type_for(tag_name: str) -> BlockType:
    """把标签名映射成块类型。"""
    if tag_name in _HEADING_LEVELS:
        return BlockType.HEADING
    if tag_name == "p":
        return BlockType.PARAGRAPH
    if tag_name == "li":
        return BlockType.LIST_ITEM
    return BlockType.TABLE


def extract_blocks(html: str, selector: str) -> list[Block]:
    """从 HTML 中提取内容块。

    只在 selector 选中的容器内提取；容器之外的导航、页脚、语言选择器
    一律不进结果。

    `<h1>` 通常位于容器之外（它是页面标题而非正文），因此一般不会
    出现在返回值里——文档标题由调用方另行提取。
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_REMOVED_TAGS):
        tag.decompose()

    container = soup.select_one(selector)
    if container is None:
        raise HtmlParseError("未找到正文容器：%s" % selector)

    # 嵌套标签若各自提取会让同一段文字出现两次（如 <li><p>…</p></li>），
    # 因此只保留最外层的那一个
    found = container.find_all(_EXTRACT_TAGS)
    extracted_ids = {id(element) for element in found}
    outermost = [
        element
        for element in found
        if not any(id(parent) in extracted_ids for parent in element.parents)
    ]

    blocks: list[Block] = []
    for element in outermost:
        tag_name = element.name
        # 行内元素（<a>、<b>、<span>）边界之间不能插空格：源码里本来没有，
        # 插了会让中文标点前多出空格（<a>卖家平台</a>。 变成「卖家平台 。」）。
        # 文本节点自身带的空格会自然保留，英文词间不会粘连。
        # 表格相反，单元格之间必须分隔，否则相邻格的内容会粘成一个词。
        separator = " " if tag_name == "table" else ""
        text = normalise_whitespace(element.get_text(separator, strip=False))
        if not text:
            continue
        blocks.append(
            Block(
                type=_block_type_for(tag_name),
                text=text,
                order=len(blocks),
                level=_HEADING_LEVELS.get(tag_name, 0),
            )
        )
    return blocks


def _extract_title(soup: BeautifulSoup) -> str:
    """取文档标题。

    优先用 `<h1>`——它是页面的主标题，通常位于正文容器之外，
    因此不会出现在 extract_blocks 的产出里。没有 `<h1>` 时退回 `<title>`。
    """
    heading = soup.find("h1")
    if heading is not None:
        text = normalise_whitespace(heading.get_text("", strip=False))
        if text:
            return text

    if soup.title is not None:
        text = normalise_whitespace(soup.title.get_text("", strip=False))
        if text:
            return text

    raise HtmlParseError("未能提取文档标题：页面既没有 <h1> 也没有 <title>")


def parse_html(
    raw_path: Path,
    doc_id: str,
    source_url: str,
    *,
    parsed_at: datetime | None = None,
) -> ParsedDocument:
    """把 HTML 文件解析成 ParsedDocument。

    source_url 用于查该站点的正文容器选择器，未配置的站点直接报错。
    parsed_at 可注入，便于测试固定时间戳。
    """
    # 先判绝对路径再判存在性：相对路径会随运行目录变化而失效，
    # 若先报「文件不存在」，排查方向会被带偏（见 CLAUDE.md 5.2）
    if not raw_path.is_absolute():
        raise HtmlParseError("raw_path 必须是绝对路径，收到：%s" % raw_path)
    if not raw_path.is_file():
        raise HtmlParseError("原始文件不存在：%s" % raw_path)

    html = raw_path.read_text(encoding="utf-8", errors="replace")

    selector = content_selector_for(source_url)
    blocks = extract_blocks(html, selector)
    if not blocks:
        raise HtmlParseError(
            "正文容器内没有解析出任何内容：%s。请确认选择器是否仍然有效"
            % selector
        )

    # 为取标题再解析一遍 HTML。单页通常几百 KB，这点开销换来的是
    # extract_blocks 与 _extract_title 各自独立、边界清晰。
    title = _extract_title(BeautifulSoup(html, "html.parser"))

    return ParsedDocument(
        doc_id=doc_id,
        title=title,
        source_format=SourceFormat.HTML,
        source_path=raw_path,
        parser_version=PARSER_VERSION,
        parsed_at=parsed_at if parsed_at is not None else datetime.now(),
        blocks=blocks,
    )
