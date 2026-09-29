"""导入脚本的单元测试。

只测报告渲染这一段——它是纯函数，且报告是使用者唯一会看的输出，
少一段或者把话说不明白，比流程出错更难发现。

脚本的其余部分是运维入口，依赖真实服务，不在这里测。
"""

from __future__ import annotations

from cbe_rag.indexing.models import DocumentOutcome, ImportAction, ImportReport
from cbe_rag.storage.records import DocumentRecord

from datetime import date

from import_corpus import render_report

TODAY = date(2026, 9, 29)


def make_outcome(**overrides: object) -> DocumentOutcome:
    """造一份文档的处理结果。"""
    fields: dict[str, object] = {
        "file_name": "a.html",
        "action": ImportAction.NEW,
        "ok": True,
        "detail": "入库 1 个父块、2 个子块",
    }
    fields.update(overrides)
    return DocumentOutcome(**fields)  # type: ignore[arg-type]


def make_orphan() -> DocumentRecord:
    """造一条「库里有、清单里没有」的记录。"""
    return DocumentRecord(
        doc_id="doc-1",
        content_hash="a" * 64,
        status="indexed",  # type: ignore[arg-type]
        title="旧版增值税指南",
        platform="amazon",
        source_url="https://example.org/gone",
        publisher="amazon",
        country="EU",
        doc_type="guideline",
        effective_date=None,
        collected_date=TODAY,
        raw_path="C:/data/raw/gone.html",
    )


def make_report(**overrides: object) -> ImportReport:
    """造一份导入报告。"""
    fields: dict[str, object] = {
        "outcomes": [],
        "missing_files": [],
        "orphans": [],
    }
    fields.update(overrides)
    return ImportReport(**fields)  # type: ignore[arg-type]


class TestOutcomeLines:
    def test_prints_one_line_per_document(self) -> None:
        report = make_report(
            outcomes=[make_outcome(file_name="a.html"), make_outcome(file_name="b.html")]
        )

        text = "\n".join(render_report(report))

        assert "a.html" in text
        assert "b.html" in text

    def test_marks_success_and_failure_differently(self) -> None:
        # 一份失败夹在一堆成功里，要能一眼扫出来
        report = make_report(
            outcomes=[
                make_outcome(),
                make_outcome(file_name="bad.bin", ok=False, detail="ParseRouteError"),
            ]
        )

        text = "\n".join(render_report(report))

        assert "OK" in text
        assert "FAIL" in text

    def test_keeps_the_detail_text(self) -> None:
        # detail 是排查的线索，不能被渲染过程吃掉
        report = make_report(outcomes=[make_outcome(detail="元数据不齐（缺 publisher）")])

        text = "\n".join(render_report(report))

        assert "publisher" in text


class TestActionCounts:
    def test_counts_each_action(self) -> None:
        report = make_report(
            outcomes=[
                make_outcome(action=ImportAction.NEW),
                make_outcome(action=ImportAction.NEW),
                make_outcome(action=ImportAction.SKIP),
            ]
        )

        text = "\n".join(render_report(report))

        assert "新建 2" in text
        assert "跳过 1" in text

    def test_zero_counts_still_printed(self) -> None:
        # 一项都没有时也要显示 0，否则会让人以为这一栏漏了
        text = "\n".join(render_report(make_report()))

        assert "重跑 0" in text


class TestMissingFiles:
    def test_lists_each_missing_file_with_its_line_number(self) -> None:
        # 清单是手工维护的，报行号才找得到该改哪一行
        from cbe_rag.ingestion.fetcher.collector import MissingFile
        from pathlib import Path

        report = make_report(
            missing_files=[
                MissingFile(
                    line_number=4,
                    file_name="absent.html",
                    expected_path=Path("C:/data/raw/absent.html"),
                )
            ]
        )

        text = "\n".join(render_report(report))

        assert "第 4 行" in text
        assert "absent.html" in text


class TestOrphans:
    def test_lists_orphans(self) -> None:
        report = make_report(orphans=[make_orphan()])

        text = "\n".join(render_report(report))

        assert "旧版增值税指南" in text
        assert "gone.html" in text

    def test_says_it_will_not_act_on_its_own(self) -> None:
        # 这段话是给使用者看的：库里这些文档还在被检索，但程序不会
        # 自作主张删掉它们
        report = make_report(orphans=[make_orphan()])

        text = "\n".join(render_report(report))

        assert "不替你做这个判断" in text

    def test_no_note_when_there_are_no_orphans(self) -> None:
        # 没有孤儿还挂着那句提醒，会让人以为漏看了什么
        text = "\n".join(render_report(make_report()))

        assert "不替你做这个判断" not in text

    def test_section_is_always_present(self) -> None:
        # 这一段的标题必须一直在：没有孤儿是正常情况，
        # 但「有没有」本身是使用者关心的
        text = "\n".join(render_report(make_report()))

        assert "库里有、清单里没有（0 份）" in text


class TestEmptyReport:
    def test_renders_without_error(self) -> None:
        lines = render_report(make_report())

        assert lines
        assert any("逐份处理结果（0 份）" in line for line in lines)
