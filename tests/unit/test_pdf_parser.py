"""PDF 解析器的单元测试。

测试用的 PDF 由 PyMuPDF 现场生成，不依赖外部文件：
这样测试在任何机器上都能跑，语料变动也不会影响它。
真实语料的验证另做。
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from cbe_rag.ingestion.parser.pdf_parser import (
    PdfParseError,
    extract_lines,
)

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
