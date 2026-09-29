"""批量导入的单元测试。

存储与嵌入用假实现，不连服务；清单与原始文件是临时目录里的真文件——
「清单里有什么、文件在不在」正是这一步的输入。

假对象与构造辅助见 indexing_fakes.py。
"""

from __future__ import annotations

from pathlib import Path

from cbe_rag.indexing.models import ImportAction
from cbe_rag.indexing.service import import_documents
from cbe_rag.storage.ddl import DocumentStatus
from indexing_fakes import (
    HTML_WITH_CONTENT,
    TODAY,
    RecordingMysqlStore,
    make_context,
    make_record,
    manifest_row,
    write_manifest,
)

# 文件头既不是 PDF 也不是 HTML，判类型那一关就过不去
UNRECOGNISABLE = b"\x00\x01\x02 not a document"


def make_env(tmp_path: Path) -> tuple[Path, Path]:
    """建好 raw 目录并写一个能解析的 HTML，返回 (清单路径, raw 目录)。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "a.html").write_text(HTML_WITH_CONTENT, encoding="utf-8")
    return write_manifest(tmp_path, [manifest_row("a.html")]), raw_dir


def make_raw_dir(tmp_path: Path, *files: str) -> Path:
    """建一个 raw 目录，按文件名写入能解析的 HTML。"""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for name in files:
        (raw_dir / name).write_text(HTML_WITH_CONTENT, encoding="utf-8")
    return raw_dir


class TestBatch:
    def test_processes_every_listed_document(self, tmp_path: Path) -> None:
        manifest, raw_dir = make_env(tmp_path)

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert len(report.outcomes) == 1
        assert report.outcomes[0].action is ImportAction.NEW

    def test_processes_documents_in_manifest_order(self, tmp_path: Path) -> None:
        raw_dir = make_raw_dir(tmp_path, "a.html", "b.html")
        manifest = write_manifest(
            tmp_path, [manifest_row("a.html"), manifest_row("b.html")]
        )

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert [outcome.file_name for outcome in report.outcomes] == [
            "a.html",
            "b.html",
        ]

    def test_reports_missing_files_separately(self, tmp_path: Path) -> None:
        # 缺文件的条目不该混进 outcomes，两者要能分开看
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        manifest = write_manifest(
            tmp_path, [manifest_row("a.html"), manifest_row("b.html")]
        )

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert [item.file_name for item in report.missing_files] == [
            "a.html",
            "b.html",
        ]
        assert report.outcomes == []


class TestFailureIsolation:
    def make_broken_env(self, tmp_path: Path) -> tuple[Path, Path]:
        raw_dir = make_raw_dir(tmp_path, "good.html")
        (raw_dir / "bad.bin").write_bytes(UNRECOGNISABLE)
        manifest = write_manifest(
            tmp_path, [manifest_row("good.html"), manifest_row("bad.bin")]
        )
        return manifest, raw_dir

    def test_one_failure_does_not_stop_the_batch(self, tmp_path: Path) -> None:
        # 失败通常是个别文件的问题，没有理由让剩下的都不处理
        manifest, raw_dir = self.make_broken_env(tmp_path)

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert [outcome.ok for outcome in report.outcomes] == [True, False]

    def test_failed_outcome_names_the_exception(self, tmp_path: Path) -> None:
        # 报告里只说「失败」对排查没有帮助，最有用的是它是什么错
        manifest, raw_dir = self.make_broken_env(tmp_path)

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert "Error" in report.outcomes[1].detail

    def test_failed_outcome_carries_no_action(self, tmp_path: Path) -> None:
        # 异常可能发生在判重之前，那时确实不知道该走哪条路，
        # 编一个动作填上只会让报告读起来更糊涂
        manifest, raw_dir = self.make_broken_env(tmp_path)

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert report.outcomes[1].action is None

    def test_successful_neighbour_is_unaffected(self, tmp_path: Path) -> None:
        manifest, raw_dir = self.make_broken_env(tmp_path)
        context = make_context()

        import_documents(manifest, raw_dir, context, today=TODAY)

        # 两份都登记了——登记发生在解析之前，坏文件也会留下记录，
        # 这样才查得到「这份收进来了但没跑通」。只有一份真正入库。
        assert len(context.mysql.inserted) == 2
        assert len(context.mysql.chunks_written) == 1
        assert len(context.milvus.upserted) == 1


class TestOrphans:
    def test_reports_documents_absent_from_the_manifest(self, tmp_path: Path) -> None:
        # 库里有、清单里没有
        manifest, raw_dir = make_env(tmp_path)
        stale = make_record("b" * 64, raw_path="C:/data/raw/removed.html")
        context = make_context(mysql=RecordingMysqlStore(active=[stale]))

        report = import_documents(manifest, raw_dir, context, today=TODAY)

        assert report.orphans == [stale]

    def test_listed_documents_are_not_orphans(self, tmp_path: Path) -> None:
        manifest, raw_dir = make_env(tmp_path)
        current = make_record("b" * 64, raw_path=str(raw_dir / "a.html"))
        context = make_context(mysql=RecordingMysqlStore(active=[current]))

        report = import_documents(manifest, raw_dir, context, today=TODAY)

        assert report.orphans == []

    def test_listed_but_missing_files_are_not_orphans(self, tmp_path: Path) -> None:
        # 清单里这一行还在，只是文件丢了。报成孤儿会误导人以为该删这一行。
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        manifest = write_manifest(tmp_path, [manifest_row("absent.html")])
        current = make_record("b" * 64, raw_path=str(raw_dir / "absent.html"))
        context = make_context(mysql=RecordingMysqlStore(active=[current]))

        report = import_documents(manifest, raw_dir, context, today=TODAY)

        assert report.missing_files
        assert report.orphans == []

    def test_empty_store_means_no_orphans(self, tmp_path: Path) -> None:
        manifest, raw_dir = make_env(tmp_path)

        report = import_documents(manifest, raw_dir, make_context(), today=TODAY)

        assert report.orphans == []

    def test_orphans_are_not_taken_down(self, tmp_path: Path) -> None:
        # 删除不可逆，而且清单改错一个字就会让文档从检索里消失，
        # 因此只报告，由人看过清单再决定
        manifest, raw_dir = make_env(tmp_path)
        stale = make_record("b" * 64, raw_path="C:/data/raw/removed.html")
        mysql = RecordingMysqlStore(active=[stale])

        import_documents(manifest, raw_dir, make_context(mysql=mysql), today=TODAY)

        assert not any(
            status is DocumentStatus.SUPERSEDED
            for _, status in mysql.status_updates
        )
