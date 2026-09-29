"""解析路由的单元测试。

路由串起四件事：判类型、探难度、组装链、逐层尝试。重点验证
**链长由难度决定**，以及失败时记录是否足够人工排查。
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from cbe_rag.ingestion.parser.difficulty import (
    DifficultyLevel,
    DifficultyReport,
)
from cbe_rag.ingestion.parser.router import (
    ParseRouteError,
    build_chain,
    detect_format,
    parse_document,
)
from cbe_rag.ingestion.parser.schema import SourceFormat
from cbe_rag.ingestion.parser.tier import ParseRequest

AMAZON_URL = "https://sellercentral.amazon.com/help/hub/reference/external/G202163020"

# 一份内容充实的页面：能通过质量评估。
# 每个问题的答案必须不同——内容重复会被质量评估按重复率拦下，
# 那是它的职责，不是测试数据的容错空间。
RICH_HTML = (
    "<html><body><h1>欧洲增值税常见问题</h1>"
    "<div id='help-content'>"
    + "".join(
        "<h4>第 %d 个问题</h4><p>这是第 %d 个问题的答案，内容要足够长才能通过字符数下限，"
        "因此这里多写一些文字凑够长度，同时保证每段内容互不相同。</p>" % (i, i)
        for i in range(5)
    )
    + "</div></body></html>"
)


def write_file(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def make_pdf(tmp_path: Path, *, text: str = "A" * 400, pages: int = 3) -> Path:
    """生成测试 PDF。

    两点都踩过：每页文字必须不同（内容相同的页会被质量评估按重复率
    拦下），且要**短到不超页宽**——超出的部分会被 PDF 截断，
    后缀一丢三页就变得一模一样了。
    """
    document = fitz.open()
    for index in range(pages):
        page = document.new_page(width=595, height=842)
        if text:
            line = ("Page %d %s" % (index, text))[:120]
            page.insert_text((72, 200), line, fontsize=12)
    path = tmp_path / "sample.pdf"
    document.save(str(path))
    document.close()
    return path


class TestDetectFormat:
    def test_pdf_is_recognised_by_magic_bytes(self, tmp_path: Path) -> None:
        assert detect_format(make_pdf(tmp_path)) is SourceFormat.PDF

    def test_html_is_recognised(self, tmp_path: Path) -> None:
        path = write_file(tmp_path, "x.html", RICH_HTML)

        assert detect_format(path) is SourceFormat.HTML

    def test_content_wins_over_extension(self, tmp_path: Path) -> None:
        # 下载的文件常被改过名。把 PDF 当 HTML 解析产出的是乱码而不是
        # 异常——猜错比报错难排查，所以按内容判。
        path = make_pdf(tmp_path)
        misnamed = path.with_name("actually_a_pdf.html")
        path.rename(misnamed)

        assert detect_format(misnamed) is SourceFormat.PDF

    def test_html_with_bom_is_recognised(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.html"
        path.write_text("﻿" + RICH_HTML, encoding="utf-8")

        assert detect_format(path) is SourceFormat.HTML

    def test_unknown_content_is_rejected(self, tmp_path: Path) -> None:
        path = write_file(tmp_path, "x.docx", "PK\x03\x04 这是压缩包，不是 HTML")

        with pytest.raises(ParseRouteError, match="无法识别"):
            detect_format(path)

    def test_error_names_the_extension(self, tmp_path: Path) -> None:
        path = write_file(tmp_path, "x.docx", "PK\x03\x04 压缩包")

        with pytest.raises(ParseRouteError) as excinfo:
            detect_format(path)

        assert ".docx" in str(excinfo.value)

    def test_relative_path_is_rejected(self) -> None:
        with pytest.raises(ParseRouteError, match="绝对路径"):
            detect_format(Path("data/raw/x.html"))

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ParseRouteError, match="不存在"):
            detect_format(tmp_path / "absent.html")


class TestBuildChain:
    def test_simple_document_gets_one_tier(self) -> None:
        # 这就是「简单文档不用很多层」——不为它准备降级层
        simple = DifficultyReport(DifficultyLevel.SIMPLE, [])

        chain = build_chain(SourceFormat.HTML, simple)

        assert len(chain) == 1
        assert chain[0].name == "html.selector"

    def test_hard_document_gets_fallback_tiers(self) -> None:
        hard = DifficultyReport(DifficultyLevel.HARD, [])

        chain = build_chain(SourceFormat.HTML, hard)

        assert [tier.name for tier in chain] == ["html.selector", "html.generic"]

    def test_primary_tier_comes_first(self) -> None:
        # 主力在前：绝大多数文档靠它就能过，降级层不该被优先尝试
        hard = DifficultyReport(DifficultyLevel.HARD, [])

        for fmt in (SourceFormat.HTML, SourceFormat.PDF):
            chain = build_chain(fmt, hard)
            assert chain[0].name.endswith(("selector", "text_layer"))


class TestParseDocument:
    def test_simple_document_passes_on_first_tier(self, tmp_path: Path) -> None:
        path = write_file(tmp_path, "x.html", RICH_HTML)
        request = ParseRequest("d", "t", AMAZON_URL, path)

        outcome = parse_document(request)

        assert outcome.document is not None
        assert outcome.needs_manual is False
        assert [a.tier for a in outcome.attempts] == ["html.selector"]
        assert outcome.difficulty == "simple"

    def test_hard_document_has_both_attempts_recorded(self, tmp_path: Path) -> None:
        # 站点没配选择器 → 判为困难 → 走两层
        path = write_file(tmp_path, "x.html", RICH_HTML)
        request = ParseRequest("d", "t", "https://example.com/x", path)

        outcome = parse_document(request)

        assert outcome.difficulty == "hard"
        assert len(outcome.attempts) >= 2

    def test_failure_detail_explains_why(self, tmp_path: Path) -> None:
        # 全部失败时，这份记录是人工处理的唯一线索
        path = write_file(tmp_path, "x.html", "<html><body><p>x</p></body></html>")
        request = ParseRequest("d", "t", "https://example.com/x", path)

        outcome = parse_document(request)

        assert outcome.needs_manual is True
        assert all(a.detail for a in outcome.attempts), "每条尝试都要写清原因"

    def test_too_thin_document_needs_manual(self, tmp_path: Path) -> None:
        thin = "<html><body><h1>标</h1><div id='help-content'><p>短</p></div></body></html>"
        path = write_file(tmp_path, "x.html", thin)
        request = ParseRequest("d", "t", AMAZON_URL, path)

        outcome = parse_document(request)

        assert outcome.needs_manual is True

    def test_pdf_goes_through_the_pdf_tier(self, tmp_path: Path) -> None:
        request = ParseRequest("d", "t", None, make_pdf(tmp_path))

        outcome = parse_document(request)

        assert outcome.document is not None
        assert outcome.attempts[0].tier == "pdf.text_layer"


class TestPackageExports:
    """包入口的导出完整性。

    回归：__all__ 里列了名字但 import 语句忘了加，调用方从包入口
    取不到、必须绕到子模块——这种不一致靠肉眼看不出来。
    """

    def test_all_declared_names_are_importable(self) -> None:
        import cbe_rag.ingestion.parser as package

        missing = [name for name in package.__all__ if not hasattr(package, name)]

        assert not missing, "导出清单里有取不到的名字：%s" % missing

    def test_router_entries_are_reachable_from_package(self) -> None:
        # 日常调用走路由入口，这几个必须能从包入口直接取到
        from cbe_rag.ingestion.parser import (
            ParseRequest,
            assess,
            parse_document,
            probe_difficulty,
        )

        assert all(callable(x) for x in (parse_document, assess, probe_difficulty))
        assert ParseRequest is not None
