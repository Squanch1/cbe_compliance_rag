"""PDF 解析器的单元测试。

测试用的 PDF 由 PyMuPDF 现场生成，不依赖外部文件：
这样测试在任何机器上都能跑，语料变动也不会影响它。
真实语料的验证另做。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import fitz
import pytest

from cbe_rag.ingestion.parser.pdf_parser import (
    PdfParseError,
    TextLine,
    body_font_size,
    build_blocks,
    extract_lines,
    heading_levels,
    parse_pdf,
)
from cbe_rag.ingestion.parser.schema import BlockType, SourceFormat

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
BODY_SIZE = 12.0
HEADING_SIZE = 14.0
PAGE_NUMBER_SIZE = 8.0


def make_pdf(
    tmp_path: Path,
    pages: list[list[tuple[str, float, float, float]]],
    name: str = "sample.pdf",
) -> Path:
    """生成测试 PDF。

    pages 每项是一页，元素为 (文本, 字号, x, y)。
    """
    document = fitz.open()
    for items in pages:
        page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        for text, size, x, y in items:
            page.insert_text((x, y), text, fontsize=size)
    path = tmp_path / name
    document.save(str(path))
    document.close()
    return path


def body_line(text: str, y: float = 200.0) -> tuple[str, float, float, float]:
    return (text, BODY_SIZE, 72.0, y)


def heading_line(text: str, y: float = 100.0) -> tuple[str, float, float, float]:
    return (text, HEADING_SIZE, 72.0, y)


def page_number(text: str) -> tuple[str, float, float, float]:
    # 与真实语料一致：8pt、页面底部居中
    return (text, PAGE_NUMBER_SIZE, 288.0, 790.0)


class TestExtractLines:
    def test_extracts_single_line(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("Import One-Stop Shop")]])

        lines = extract_lines(path)

        assert len(lines) == 1
        assert lines[0].text == "Import One-Stop Shop"

    def test_records_font_size(self, tmp_path: Path) -> None:
        path = make_pdf(
            tmp_path, [[heading_line("Chapter 1"), body_line("Body text")]]
        )

        sizes = {line.text: line.size for line in extract_lines(path)}

        assert sizes["Chapter 1"] == HEADING_SIZE
        assert sizes["Body text"] == BODY_SIZE

    def test_page_number_starts_at_one(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("First page")], [body_line("Second")]])

        pages = [line.page for line in extract_lines(path)]

        assert pages == [1, 2]

    def test_empty_page_yields_nothing(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("has content")], []])

        assert len(extract_lines(path)) == 1

    def test_top_coordinate_is_recorded(self, tmp_path: Path) -> None:
        path = make_pdf(
            tmp_path, [[body_line("upper", y=150.0), body_line("lower", y=300.0)]]
        )

        tops = [line.top for line in extract_lines(path)]

        assert tops[0] < tops[1]


class TestPageNumberRemoval:
    def test_page_number_is_removed(self, tmp_path: Path) -> None:
        path = make_pdf(
            tmp_path,
            [
                [body_line("first page body"), page_number("1/2")],
                [body_line("second page body"), page_number("2/2")],
            ],
        )

        texts = [line.text for line in extract_lines(path)]

        assert texts == ["first page body", "second page body"]

    def test_number_pattern_not_at_bottom_is_kept(self, tmp_path: Path) -> None:
        # 正文里出现「1/2」这种分数是正常的，不能凭模式就删
        path = make_pdf(tmp_path, [[("1/2", BODY_SIZE, 72.0, 400.0)]])

        assert [line.text for line in extract_lines(path)] == ["1/2"]

    def test_bottom_text_not_matching_pattern_is_kept(self, tmp_path: Path) -> None:
        # 正文可以延伸到页面底部，实测欧盟文档就有
        path = make_pdf(tmp_path, [[("From 1 July 2028, the scope is extended", BODY_SIZE, 72.0, 790.0)]])

        assert len(extract_lines(path)) == 1

    def test_removal_requires_both_position_and_pattern(self, tmp_path: Path) -> None:
        # 底部 + 非页码格式 → 保留；非底部 + 页码格式 → 保留
        path = make_pdf(
            tmp_path,
            [[
                ("Article 39", BODY_SIZE, 72.0, 800.0),
                ("3/7", BODY_SIZE, 72.0, 400.0),
                ("3/7", PAGE_NUMBER_SIZE, 288.0, 795.0),
            ]],
        )

        assert [line.text for line in extract_lines(path)] == ["Article 39", "3/7"]


class TestInputValidation:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PdfParseError, match="不存在"):
            extract_lines(tmp_path / "absent.pdf")

    def test_relative_path_raises(self) -> None:
        # 相对路径会随运行目录变化而失效，报错要说准原因
        with pytest.raises(PdfParseError, match="绝对路径"):
            extract_lines(Path("data/raw/x.pdf"))

    def test_non_pdf_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "fake.pdf"
        path.write_text("这不是 PDF", encoding="utf-8")

        with pytest.raises(PdfParseError):
            extract_lines(path)


def lines_with(*specs: tuple[str, float]) -> list[TextLine]:
    """按 (文本, 字号) 批量造行，y 坐标依次递增。"""
    return [
        TextLine(page=1, text=text, size=size, top=100.0 + index * 20.0)
        for index, (text, size) in enumerate(specs)
    ]


class TestBodyFontSize:
    def test_picks_the_most_common_size(self) -> None:
        lines = lines_with(
            ("正文一", 12.0), ("正文二", 12.0), ("正文三", 12.0),
            ("标题", 16.0), ("小字", 9.0),
        )

        assert body_font_size(lines) == 12.0

    def test_single_size_document(self) -> None:
        assert body_font_size(lines_with(("只有正文", 11.0))) == 11.0

    def test_empty_input_raises(self) -> None:
        # 没有行就没有「正文」，静默返回一个默认字号会让后续判断全错
        with pytest.raises(PdfParseError, match="统计"):
            body_font_size([])


class TestHeadingLevels:
    def test_larger_sizes_become_headings(self) -> None:
        lines = lines_with(("正文", 12.0), ("二级", 13.0), ("一级", 14.0))

        levels = heading_levels(lines)

        assert levels == {14.0: 1, 13.0: 2}

    def test_levels_are_relative_not_hardcoded(self) -> None:
        # 换一份排版不同的文档，层级应随字号相对关系变化
        lines = lines_with(("正文", 10.0), ("大标题", 20.0), ("小标题", 15.0))

        assert heading_levels(lines) == {20.0: 1, 15.0: 2}

    def test_no_larger_size_yields_empty_mapping(self) -> None:
        # 全文同一字号：没有标题，不该硬造出层级
        assert heading_levels(lines_with(("甲", 12.0), ("乙", 12.0))) == {}

    def test_sizes_below_body_are_ignored(self) -> None:
        # 11pt 是次级正文（免责声明等），不是标题。
        # 正文给足行数，避免落到「每档一行」的平局分支上去。
        lines = lines_with(
            ("正文一", 12.0), ("正文二", 12.0), ("正文三", 12.0),
            ("免责声明", 11.0), ("标题", 14.0),
        )

        assert heading_levels(lines) == {14.0: 1}

    def test_empty_input_yields_empty_mapping(self) -> None:
        # 空文档没有正文字号可统计，但也不该报错——空文档返回空结果是合理的
        assert heading_levels([]) == {}


def make_lines(*specs: tuple) -> list[TextLine]:
    """造行。specs 元素为 (文本, 字号, y) 或 (文本, 字号, y, 页码)。"""
    return [
        TextLine(page=spec[3] if len(spec) > 3 else 1, text=spec[0], size=spec[1], top=spec[2])
        for spec in specs
    ]


class TestBuildBlocks:
    def test_lines_in_one_paragraph_are_merged(self) -> None:
        # 行距正常（13.8pt）的行属于同一段
        lines = make_lines(("第一行", 12.0, 100.0), ("第二行", 12.0, 113.8), ("第三行", 12.0, 127.6))

        blocks = build_blocks(lines)

        assert len(blocks) == 1
        assert blocks[0].text == "第一行 第二行 第三行"

    def test_large_gap_starts_new_paragraph(self) -> None:
        # 段落间距约 25.8pt，明显大于正常行距
        lines = make_lines(("上段", 12.0, 100.0), ("下段", 12.0, 125.8))

        blocks = build_blocks(lines)

        assert [b.text for b in blocks] == ["上段", "下段"]

    def test_cross_page_breaks_the_paragraph(self) -> None:
        # 跨页一律断开：判断跨页续段需要额外信息，误合的代价更大
        lines = make_lines(("页底", 12.0, 780.0, 1), ("页顶", 12.0, 70.0, 2))

        blocks = build_blocks(lines)

        assert [b.text for b in blocks] == ["页底", "页顶"]

    def test_heading_becomes_its_own_block_with_level(self) -> None:
        lines = make_lines(("1.1 INTRODUCTION", 13.0, 100.0), ("正文", 12.0, 120.0))

        blocks = build_blocks(lines)

        assert blocks[0].type is BlockType.HEADING
        assert blocks[0].level == 1
        assert blocks[0].text == "1.1 INTRODUCTION"

    def test_body_after_heading_is_a_separate_paragraph(self) -> None:
        lines = make_lines(
            ("标题", 14.0, 100.0), ("第一行", 12.0, 120.0), ("第二行", 12.0, 133.8)
        )

        blocks = build_blocks(lines)

        assert [b.type for b in blocks] == [BlockType.HEADING, BlockType.PARAGRAPH]
        assert blocks[1].text == "第一行 第二行"

    def test_same_y_blocks_are_joined(self) -> None:
        # 目录里的章节号与标题在同一行、属于不同块
        lines = make_lines(("1", 14.0, 100.0), ("KEY ELEMENTS", 14.0, 100.0))

        blocks = build_blocks(lines)

        assert len(blocks) == 1
        assert blocks[0].text == "1 KEY ELEMENTS"

    def test_order_is_contiguous(self) -> None:
        lines = make_lines(
            ("标题", 14.0, 100.0), ("段一", 12.0, 120.0), ("段二", 12.0, 150.0)
        )

        assert [b.order for b in build_blocks(lines)] == [0, 1, 2]

    def test_empty_input_yields_no_blocks(self) -> None:
        assert build_blocks([]) == []

    def test_document_without_headings_is_all_paragraphs(self) -> None:
        lines = make_lines(("甲", 12.0, 100.0), ("乙", 12.0, 113.8))

        assert all(b.type is BlockType.PARAGRAPH for b in build_blocks(lines))

    def test_tie_breaks_towards_the_smaller_size(self) -> None:
        # 标题与正文各一行时是平局。标题总比正文大，因此取较小的那个。
        # 用 most_common 会在平局时按出现顺序任选，结果随排版而变。
        lines = make_lines(("标题", 13.0, 100.0), ("正文", 12.0, 120.0))

        assert body_font_size(lines) == 12.0

    def test_tie_break_does_not_depend_on_line_order(self) -> None:
        forward = make_lines(("标题", 13.0, 100.0), ("正文", 12.0, 120.0))
        backward = make_lines(("正文", 12.0, 100.0), ("标题", 13.0, 120.0))

        assert body_font_size(forward) == body_font_size(backward) == 12.0


class TestParsePdf:
    def test_returns_pdf_source_format(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("Body text")]])

        document = parse_pdf(path, "doc-id", "手工标题")

        assert document.source_format is SourceFormat.PDF

    def test_title_comes_from_caller(self, tmp_path: Path) -> None:
        # PDF 没有 <h1> 等价物，封面标题常跨多个字号，猜出来的不完整，
        # 因此标题由调用方传入（通常来自清单里的人工登记）
        path = make_pdf(tmp_path, [[heading_line("Explanatory Notes")]])

        assert parse_pdf(path, "doc-id", "手工标题").title == "手工标题"

    def test_blocks_match_build_blocks(self, tmp_path: Path) -> None:
        path = make_pdf(
            tmp_path,
            [[heading_line("Chapter 1"), body_line("Body", y=120.0)]],
        )

        document = parse_pdf(path, "doc-id", "标题")

        assert [b.type for b in document.blocks] == [BlockType.HEADING, BlockType.PARAGRAPH]

    def test_identity_fields_are_carried(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("Body")]])

        document = parse_pdf(path, "doc-id-123", "标题")

        assert document.doc_id == "doc-id-123"
        assert document.source_path == path
        assert document.parser_version

    def test_parsed_at_is_injectable(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, [[body_line("Body")]])
        moment = datetime(2026, 9, 28, 12, 0, 0)

        assert parse_pdf(path, "id", "标题", parsed_at=moment).parsed_at == moment

    def test_pdf_without_text_raises(self, tmp_path: Path) -> None:
        # 扫描件这类没有文本层的 PDF 应明确报错，而不是产出空文档
        path = make_pdf(tmp_path, [[]])

        with pytest.raises(PdfParseError, match="没有提取到"):
            parse_pdf(path, "id", "标题")

    def test_relative_path_raises(self) -> None:
        with pytest.raises(PdfParseError, match="绝对路径"):
            parse_pdf(Path("data/raw/x.pdf"), "id", "标题")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PdfParseError, match="不存在"):
            parse_pdf(tmp_path / "absent.pdf", "id", "标题")
