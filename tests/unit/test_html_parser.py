"""HTML 解析器的单元测试。

内联的 HTML 片段仿照真实语料的结构：正文在 div#help-content 里，
噪声在 header 与容器之外。真实文件的验证另做。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cbe_rag.ingestion.parser.html_parser import (
    HtmlParseError,
    content_selector_for,
    extract_blocks,
    parse_html,
)
from cbe_rag.ingestion.parser.schema import BlockType, SourceFormat

SELECTOR = "#help-content"


def page(content: str) -> str:
    """把内容包进一份仿真的页面：正文在容器内，噪声在容器外。"""
    return (
        "<!DOCTYPE html><html><head><title>页面标题</title>"
        "<style>.x{color:red}</style></head><body>"
        "<header class='mega-menu'><h3>选择您的首选语言</h3>"
        "<ul><li>English</li><li>中文</li></ul></header>"
        "<h1>欧洲增值税常见问题</h1>"
        "<div id='help-content'>%s</div>"
        "<footer><h2>需要更多帮助？</h2></footer>"
        "<script>var x = 1;</script>"
        "</body></html>" % content
    )


class TestExtractBlocks:
    def test_heading_levels_follow_tag_names(self) -> None:
        html = page("<h2>二级</h2><h3>三级</h3><h4>四级</h4>")

        blocks = extract_blocks(html, SELECTOR)

        assert [b.level for b in blocks] == [2, 3, 4]
        assert all(b.type is BlockType.HEADING for b in blocks)

    def test_paragraph_is_extracted(self) -> None:
        blocks = extract_blocks(page("<p>这是一段正文。</p>"), SELECTOR)

        assert len(blocks) == 1
        assert blocks[0].type is BlockType.PARAGRAPH
        assert blocks[0].text == "这是一段正文。"

    def test_list_item_is_extracted(self) -> None:
        blocks = extract_blocks(page("<ul><li>第一项</li><li>第二项</li></ul>"), SELECTOR)

        assert [b.type for b in blocks] == [BlockType.LIST_ITEM, BlockType.LIST_ITEM]
        assert [b.text for b in blocks] == ["第一项", "第二项"]

    def test_table_is_extracted(self) -> None:
        html = page("<table><tr><td>国家</td><td>税率</td></tr></table>")

        blocks = extract_blocks(html, SELECTOR)

        assert len(blocks) == 1
        assert blocks[0].type is BlockType.TABLE
        assert "国家" in blocks[0].text and "税率" in blocks[0].text

    def test_order_follows_document_order(self) -> None:
        html = page("<h4>问题一</h4><p>答案一</p><h4>问题二</h4><p>答案二</p>")

        blocks = extract_blocks(html, SELECTOR)

        assert [b.order for b in blocks] == [0, 1, 2, 3]
        assert [b.text for b in blocks] == ["问题一", "答案一", "问题二", "答案二"]

    def test_whitespace_is_normalised(self) -> None:
        blocks = extract_blocks(page("<p>  多个\n\n  空白   字符  </p>"), SELECTOR)

        assert blocks[0].text == "多个 空白 字符"

    def test_inline_element_does_not_introduce_space(self) -> None:
        # 回归：get_text(" ") 会在行内元素边界插空格，把
        # <a>卖家平台</a>。 变成「卖家平台 。」，中文标点前多一个空格。
        # 源码里没有这个空格，拼接时也不该加上。
        html = page("<p>必须与<a href='#'>卖家平台</a>中的法定名称一致。</p>")

        blocks = extract_blocks(html, SELECTOR)

        assert blocks[0].text == "必须与卖家平台中的法定名称一致。"

    def test_source_spacing_between_inline_elements_is_kept(self) -> None:
        # 源码里文本节点自身带的空格要保留，英文词间不能粘连
        html = page("<p><span>Hello</span> <span>World</span></p>")

        assert extract_blocks(html, SELECTOR)[0].text == "Hello World"

    def test_adjacent_inline_elements_without_space_are_joined(self) -> None:
        # 源码中就没有空格，浏览器也是这样显示的
        html = page("<p><b>VAT</b>number</p>")

        assert extract_blocks(html, SELECTOR)[0].text == "VATnumber"

    def test_table_cells_are_separated(self) -> None:
        # 表格单元格之间需要分隔，否则「德国」与「19%」会粘成一个词
        html = page("<table><tr><td>德国</td><td>19%</td></tr></table>")

        assert extract_blocks(html, SELECTOR)[0].text == "德国 19%"

    def test_empty_elements_are_skipped(self) -> None:
        html = page("<p>有内容</p><p>   </p><p></p><p>也有内容</p>")

        blocks = extract_blocks(html, SELECTOR)

        assert [b.text for b in blocks] == ["有内容", "也有内容"]

    def test_order_is_contiguous_after_skipping(self) -> None:
        # 跳过空元素后 order 不能留洞，否则下游按 order 排序会出现空档
        blocks = extract_blocks(page("<p>甲</p><p></p><p>乙</p>"), SELECTOR)

        assert [b.order for b in blocks] == [0, 1]


class TestNoiseIsolation:
    def test_content_outside_container_is_ignored(self) -> None:
        # header 里的语言选择器、footer 里的「需要更多帮助」都不该进来
        blocks = extract_blocks(page("<p>正文</p>"), SELECTOR)

        texts = " ".join(b.text for b in blocks)
        assert "选择您的首选语言" not in texts
        assert "需要更多帮助" not in texts
        assert "欧洲增值税常见问题" not in texts  # h1 在容器外

    def test_script_and_style_are_removed(self) -> None:
        html = page("<script>var a=1;</script><style>.y{}</style><p>正文</p>")

        blocks = extract_blocks(html, SELECTOR)

        assert [b.text for b in blocks] == ["正文"]

    def test_nested_elements_are_not_double_counted(self) -> None:
        # <li><p>…</p></li> 若两个都提取，同一段文字会出现两次
        blocks = extract_blocks(page("<ul><li><p>列表项内容</p></li></ul>"), SELECTOR)

        assert len(blocks) == 1
        assert blocks[0].text == "列表项内容"

    def test_container_itself_not_extracted(self) -> None:
        blocks = extract_blocks(page("<p>正文</p>"), SELECTOR)

        assert all(b.type is not BlockType.HEADING or b.level > 1 for b in blocks)


class TestContainerLookup:
    def test_missing_container_raises(self) -> None:
        # 宁可报错也不产出脏数据——没有容器意味着选择器配错了
        html = "<html><body><p>没有目标容器</p></body></html>"

        with pytest.raises(HtmlParseError, match="正文容器"):
            extract_blocks(html, SELECTOR)

    def test_error_message_names_the_selector(self) -> None:
        html = "<html><body><p>x</p></body></html>"

        with pytest.raises(HtmlParseError) as excinfo:
            extract_blocks(html, "#not-exist")

        assert "#not-exist" in str(excinfo.value)

    def test_empty_container_yields_no_blocks(self) -> None:
        assert extract_blocks(page(""), SELECTOR) == []


class TestContentSelectorFor:
    def test_known_site_returns_its_selector(self) -> None:
        selector = content_selector_for(
            "https://sellercentral.amazon.com/help/hub/reference/external/G202163020"
        )

        assert selector == "#help-content"

    def test_www_prefix_is_tolerated(self) -> None:
        selector = content_selector_for("https://www.sellercentral.amazon.com/help")

        assert selector == "#help-content"

    def test_unknown_site_raises(self) -> None:
        # 未配置的站点直接报错，逼着先看真实 DOM 再接入
        with pytest.raises(HtmlParseError, match="未配置"):
            content_selector_for("https://example.com/some/page")

    def test_error_lists_known_sites(self) -> None:
        with pytest.raises(HtmlParseError) as excinfo:
            content_selector_for("https://example.com/x")

        assert "sellercentral.amazon.com" in str(excinfo.value)


class TestParseHtml:
    """从文件到 ParsedDocument 的组装。"""

    URL = "https://sellercentral.amazon.com/help/hub/reference/external/G202163020"
    DOC_ID = "6f1c2f7e-1a2b-4c3d-8e9f-0a1b2c3d4e5f"

    def write(self, tmp_path: Path, html: str) -> Path:
        path = tmp_path / "page.html"
        path.write_text(html, encoding="utf-8")
        return path

    def test_returns_html_source_format(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, page("<p>正文</p>"))

        document = parse_html(path, self.DOC_ID, self.URL)

        assert document.source_format is SourceFormat.HTML

    def test_title_comes_from_h1(self, tmp_path: Path) -> None:
        # h1 在正文容器之外，需要单独提取
        path = self.write(tmp_path, page("<p>正文</p>"))

        assert parse_html(path, self.DOC_ID, self.URL).title == "欧洲增值税常见问题"

    def test_title_falls_back_to_title_tag(self, tmp_path: Path) -> None:
        html = "<html><head><title>页面标题</title></head><body><div id='help-content'><p>x</p></div></body></html>"
        path = self.write(tmp_path, html)

        assert parse_html(path, self.DOC_ID, self.URL).title == "页面标题"

    def test_missing_title_raises(self, tmp_path: Path) -> None:
        html = "<html><body><div id='help-content'><p>x</p></div></body></html>"
        path = self.write(tmp_path, html)

        with pytest.raises(HtmlParseError, match="标题"):
            parse_html(path, self.DOC_ID, self.URL)

    def test_blocks_match_extract_blocks(self, tmp_path: Path) -> None:
        html = page("<h4>问题</h4><p>答案</p>")
        path = self.write(tmp_path, html)

        document = parse_html(path, self.DOC_ID, self.URL)

        assert [b.text for b in document.blocks] == ["问题", "答案"]

    def test_identity_fields_are_carried(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, page("<p>x</p>"))

        document = parse_html(path, self.DOC_ID, self.URL)

        assert document.doc_id == self.DOC_ID
        assert document.source_path == path
        assert document.parser_version

    def test_parsed_at_is_injectable(self, tmp_path: Path) -> None:
        # 固定时间戳，否则测试结果随运行时刻变化
        path = self.write(tmp_path, page("<p>x</p>"))
        moment = datetime(2026, 9, 28, 12, 0, 0)

        assert parse_html(path, self.DOC_ID, self.URL, parsed_at=moment).parsed_at == moment

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(HtmlParseError, match="不存在"):
            parse_html(tmp_path / "absent.html", self.DOC_ID, self.URL)

    def test_unconfigured_site_propagates(self, tmp_path: Path) -> None:
        path = self.write(tmp_path, page("<p>x</p>"))

        with pytest.raises(HtmlParseError, match="未配置"):
            parse_html(path, self.DOC_ID, "https://example.com/x")

    def test_source_path_must_be_absolute(self, tmp_path: Path) -> None:
        # ParsedDocument 会拒绝相对路径，这里确认错误能传出来
        path = self.write(tmp_path, page("<p>x</p>"))
        relative = Path("data/raw/x.html")

        with pytest.raises(Exception, match="绝对路径"):
            parse_html(relative, self.DOC_ID, self.URL)
