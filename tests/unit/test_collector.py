"""文档收集的单元测试。

覆盖清单到 DocumentMeta 的转换，以及缺文件时的处理。
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pytest

from cbe_rag.ingestion.fetcher.collector import (
    CollectionError,
    collect_documents,
)
from cbe_rag.ingestion.fetcher.manifest import MANIFEST_COLUMNS

HEADER = ",".join(MANIFEST_COLUMNS)
TODAY = date(2026, 9, 28)


def row(**overrides: str) -> str:
    """按 MANIFEST_COLUMNS 的顺序拼一行，未指定的列留空。

    与 test_manifest.py 里的同名辅助函数一致。手工数逗号容易错位，
    因此两处都用它构造行。
    """
    cells = {name: "" for name in MANIFEST_COLUMNS}
    cells["platform"] = "amazon"
    cells.update(overrides)
    return ",".join(cells[name] for name in MANIFEST_COLUMNS)


def full_row(file_name: str, **overrides: str) -> str:
    """一份元数据齐备的行。"""
    payload = {
        "file_name": file_name,
        "title": "%s 的标题" % file_name,
        "source_url": "https://europa.eu/%s" % file_name,
        "publisher": "eu_commission",
        "country": "EU",
        "doc_type": "guideline",
    }
    payload.update(overrides)
    return row(**payload)


def build_corpus(
    tmp_path: Path, rows: list[str], files: list[str]
) -> tuple[Path, Path]:
    """建一个语料目录：raw/ 下放文件，并生成对应清单。"""
    data = tmp_path / "data"
    raw = data / "raw"
    raw.mkdir(parents=True)
    for name in files:
        (raw / name).write_text("正文内容", encoding="utf-8")

    manifest = data / "manifest.csv"
    manifest.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8-sig")
    return manifest, raw


class TestAllFilesPresent:
    def test_returns_one_document_per_row(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(
            tmp_path,
            [full_row("a.html"), full_row("b.pdf")],
            ["a.html", "b.pdf"],
        )

        result = collect_documents(manifest, raw, today=TODAY)

        assert len(result.documents) == 2
        assert result.missing_files == []

    def test_raw_path_is_absolute(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(tmp_path, [full_row("a.html")], ["a.html"])

        result = collect_documents(manifest, raw, today=TODAY)

        assert result.documents[0].raw_path.is_absolute()
        assert result.documents[0].raw_path.name == "a.html"

    def test_metadata_is_carried_over(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(
            tmp_path,
            [full_row("a.html", effective_date="2021-07-01")],
            ["a.html"],
        )

        meta = collect_documents(manifest, raw, today=TODAY).documents[0].meta

        assert meta.title == "a.html 的标题"
        assert meta.country == "EU"
        assert meta.publisher == "eu_commission"
        assert meta.effective_date == date(2021, 7, 1)

    def test_collected_date_comes_from_today(self, tmp_path: Path) -> None:
        # 注入 today 是为了让测试结果不随运行日期变化
        manifest, raw = build_corpus(tmp_path, [full_row("a.html")], ["a.html"])

        meta = collect_documents(manifest, raw, today=TODAY).documents[0].meta

        assert meta.collected_date == TODAY

    def test_each_document_gets_a_distinct_id(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(
            tmp_path,
            [full_row("a.html"), full_row("b.html")],
            ["a.html", "b.html"],
        )

        ids = {d.meta.doc_id for d in collect_documents(manifest, raw, today=TODAY).documents}

        assert len(ids) == 2

    def test_incomplete_metadata_is_still_collected(self, tmp_path: Path) -> None:
        # 元数据缺失走 pending 门禁，不影响收集本身
        manifest, raw = build_corpus(
            tmp_path, [row(file_name="a.html", title="只有标题")], ["a.html"]
        )

        meta = collect_documents(manifest, raw, today=TODAY).documents[0].meta

        assert len(meta.missing_required_fields()) == 4


class TestMissingFiles:
    def test_missing_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(tmp_path, [full_row("absent.html")], [])

        result = collect_documents(manifest, raw, today=TODAY)

        assert result.documents == []
        assert len(result.missing_files) == 1

    def test_present_files_are_still_collected(self, tmp_path: Path) -> None:
        # 支持「先填行、后下文件」的分批节奏
        manifest, raw = build_corpus(
            tmp_path,
            [full_row("have.html"), full_row("absent.html")],
            ["have.html"],
        )

        result = collect_documents(manifest, raw, today=TODAY)

        assert [d.raw_path.name for d in result.documents] == ["have.html"]
        assert [m.file_name for m in result.missing_files] == ["absent.html"]

    def test_missing_entry_carries_line_number(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(
            tmp_path,
            [full_row("have.html"), full_row("absent.html")],
            ["have.html"],
        )

        missing = collect_documents(manifest, raw, today=TODAY).missing_files[0]

        assert missing.line_number == 3

    def test_missing_entry_carries_expected_path(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(tmp_path, [full_row("absent.html")], [])

        missing = collect_documents(manifest, raw, today=TODAY).missing_files[0]

        assert missing.expected_path == raw / "absent.html"

    def test_empty_manifest_yields_nothing(self, tmp_path: Path) -> None:
        manifest, raw = build_corpus(tmp_path, [], [])

        result = collect_documents(manifest, raw, today=TODAY)

        assert result.documents == []
        assert result.missing_files == []


class TestArgumentValidation:
    def test_relative_raw_dir_is_rejected(self, tmp_path: Path) -> None:
        # 相对路径会随运行目录变化而失效（见 CLAUDE.md 5.2）
        manifest, _ = build_corpus(tmp_path, [full_row("a.html")], ["a.html"])

        with pytest.raises(CollectionError, match="绝对路径"):
            collect_documents(manifest, Path("data/raw"), today=TODAY)

    def test_missing_raw_dir_is_reported(self, tmp_path: Path) -> None:
        # 目录整个不存在时，逐条报「文件缺失」会淹没真正的原因
        manifest, raw = build_corpus(tmp_path, [full_row("a.html")], ["a.html"])
        shutil.rmtree(raw)

        with pytest.raises(CollectionError, match="目录不存在"):
            collect_documents(manifest, raw, today=TODAY)

    def test_problem_in_manifest_propagates(self, tmp_path: Path) -> None:
        # 清单本身有问题时直接抛出，不做「跳过这行继续」的处理
        manifest, raw = build_corpus(tmp_path, [row(title="缺文件名")], [])

        with pytest.raises(Exception, match="file_name"):
            collect_documents(manifest, raw, today=TODAY)
