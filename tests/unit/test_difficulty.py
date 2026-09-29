"""难度探测的单元测试。

探测决定「挂几层解析器」，判错会导致两种后果：把简单文档判难则白跑
降级层，把困难文档判简单则解析失败后直接转人工。因此两个方向都要测。
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from cbe_rag.ingestion.parser.difficulty import (
    DifficultyLevel,
    probe_difficulty,
)
from cbe_rag.ingestion.parser.schema import SourceFormat
from cbe_rag.ingestion.parser.tier import ParseRequest

AMAZON_URL = "https://sellercentral.amazon.com/help/hub/reference/external/G202163020"


def make_pdf(tmp_path: Path, *, text: str, pages: int = 3) -> Path:
    """生成一份测试 PDF。text 为空则模拟无文本层的扫描件。"""
    document = fitz.open()
    for _ in range(pages):
        page = document.new_page(width=595, height=842)
        if text:
            page.insert_text((72, 200), text, fontsize=12)
    path = tmp_path / "sample.pdf"
    document.save(str(path))
    document.close()
    return path


def pdf_request(path: Path) -> ParseRequest:
    return ParseRequest(doc_id="d", title="t", source_url=None, raw_path=path)


def html_request(url: str | None) -> ParseRequest:
    return ParseRequest(
        doc_id="d",
        title="t",
        source_url=url,
        raw_path=Path("C:/proj/data/raw/x.html"),
    )


class TestPdfDifficulty:
    def test_pdf_with_text_layer_is_simple(self, tmp_path: Path) -> None:
        path = make_pdf(tmp_path, text="A" * 300)

        report = probe_difficulty(pdf_request(path), SourceFormat.PDF)

        assert report.level is DifficultyLevel.SIMPLE

    def test_pdf_without_text_layer_is_hard(self, tmp_path: Path) -> None:
        # 扫描件：需要 OCR，而首期未实现，因此判为困难
        path = make_pdf(tmp_path, text="")

        report = probe_difficulty(pdf_request(path), SourceFormat.PDF)

        assert report.level is DifficultyLevel.HARD
        assert any("扫描件" in r for r in report.reasons)

    def test_unreadable_file_is_hard(self, tmp_path: Path) -> None:
        # 打不开的文件本身就是困难，不该在探测阶段就崩掉
        path = tmp_path / "broken.pdf"
        path.write_text("这不是 PDF", encoding="utf-8")

        report = probe_difficulty(pdf_request(path), SourceFormat.PDF)

        assert report.level is DifficultyLevel.HARD

    def test_reasons_record_what_was_sampled(self, tmp_path: Path) -> None:
        # 只给一个「困难」的结论，调阈值时无从下手
        path = make_pdf(tmp_path, text="A" * 300, pages=9)

        report = probe_difficulty(pdf_request(path), SourceFormat.PDF)

        assert any("抽样" in r for r in report.reasons)

    def test_samples_across_the_document_not_just_the_front(
        self, tmp_path: Path
    ) -> None:
        # 只抽前几页会撞上封面与目录，那几页本来字符就少，会误判成扫描件
        path = make_pdf(tmp_path, text="A" * 300, pages=9)

        report = probe_difficulty(pdf_request(path), SourceFormat.PDF)

        sampled = [r for r in report.reasons if "页码" in r][0]
        assert "9" in sampled, "应当抽到末页，而不是只看开头"


class TestHtmlDifficulty:
    def test_configured_site_is_simple(self) -> None:
        report = probe_difficulty(html_request(AMAZON_URL), SourceFormat.HTML)

        assert report.level is DifficultyLevel.SIMPLE
        assert any("#help-content" in r for r in report.reasons)

    def test_unconfigured_site_is_hard(self) -> None:
        # 没配选择器就定位不了正文，只能靠通用提取兜底
        report = probe_difficulty(
            html_request("https://example.com/x"), SourceFormat.HTML
        )

        assert report.level is DifficultyLevel.HARD

    def test_missing_source_url_is_hard(self) -> None:
        report = probe_difficulty(html_request(None), SourceFormat.HTML)

        assert report.level is DifficultyLevel.HARD
        assert any("source_url" in r for r in report.reasons)


class TestIsHard:
    def test_property_matches_level(self, tmp_path: Path) -> None:
        simple = probe_difficulty(html_request(AMAZON_URL), SourceFormat.HTML)
        hard = probe_difficulty(html_request(None), SourceFormat.HTML)

        assert simple.is_hard is False
        assert hard.is_hard is True
