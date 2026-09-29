"""HTML 解析器。

把网页转成统一中间表示的 Block 列表。

正文位置按**站点配置的选择器**定位，不做通用启发式判断。原因是实测过的：
某帮助页用「排除 header/footer/nav/aside」的通用规则，仍然残留了
语言选择器、相关文章、页脚三个噪声块。产出脏数据比直接报错更难排查。
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from cbe_rag.ingestion.parser.schema import Block, BlockType

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

_WHITESPACE = re.compile(r"\s+")


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


def _normalise(text: str) -> str:
    """折叠空白。HTML 源码里的换行与缩进对正文没有意义。"""
    return _WHITESPACE.sub(" ", text).strip()


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
        text = _normalise(element.get_text(separator, strip=False))
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
