"""清单读取与校验的单元测试。

清单是手工维护的表格，最容易出的问题是「有一处不对但不知道在哪一行」，
因此报错信息必须带行号，测试也围绕这一点展开。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cbe_rag.ingestion.fetcher.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    load_manifest,
)

HEADER = ",".join(MANIFEST_COLUMNS)


def row(**overrides: str) -> str:
    """按 MANIFEST_COLUMNS 的顺序拼一行，未指定的列留空。

    手工数逗号容易错位（把 amazon 写进 effective_date 那一列），
    用这个辅助函数构造行可以避免。
    """
    cells = {name: "" for name in MANIFEST_COLUMNS}
    cells["platform"] = "amazon"
    cells.update(overrides)
    return ",".join(cells[name] for name in MANIFEST_COLUMNS)


VALID_ROW = row(
    file_name="ioss.html",
    title="IOSS 指南",
    source_url="https://europa.eu/x",
    publisher="eu_commission",
    country="EU",
    doc_type="guideline",
    effective_date="2021-07-01",
    notes="备注",
)


def write_manifest(path: Path, *lines: str) -> Path:
    """按 utf-8-sig 写入清单，与真实的 manifest.csv 一致。"""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def manifest_with(tmp_path: Path, *rows: str) -> Path:
    return write_manifest(tmp_path / "manifest.csv", HEADER, *rows)


class TestLoadSuccess:
    def test_reads_one_row(self, tmp_path: Path) -> None:
        rows = load_manifest(manifest_with(tmp_path, VALID_ROW))

        assert len(rows) == 1
        row = rows[0]
        assert row.file_name == "ioss.html"
        assert row.title == "IOSS 指南"
        assert row.country == "EU"
        assert row.effective_date == date(2021, 7, 1)
        assert row.platform == "amazon"

    def test_header_only_returns_empty_list(self, tmp_path: Path) -> None:
        assert load_manifest(write_manifest(tmp_path / "m.csv", HEADER)) == []

    def test_bom_is_stripped_from_first_column(self, tmp_path: Path) -> None:
        # utf-8-sig 的 BOM 若没被去掉，第一列会变成 "﻿file_name"
        rows = load_manifest(manifest_with(tmp_path, VALID_ROW))

        assert rows[0].file_name == "ioss.html"

    def test_line_number_points_at_the_data_row(self, tmp_path: Path) -> None:
        # 表头算第 1 行，第一条数据是第 2 行
        rows = load_manifest(manifest_with(tmp_path, VALID_ROW))

        assert rows[0].line_number == 2

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        # 表格末尾常有多余空行
        path = write_manifest(tmp_path / "m.csv", HEADER, VALID_ROW, "", "   ")

        assert len(load_manifest(path)) == 1

    def test_optional_fields_may_be_empty(self, tmp_path: Path) -> None:
        rows = load_manifest(
            manifest_with(tmp_path, row(file_name="a.html", title="标题"))
        )

        assert rows[0].source_url is None
        assert rows[0].publisher is None
        assert rows[0].country is None
        assert rows[0].doc_type is None
        assert rows[0].effective_date is None
        assert rows[0].notes is None

    def test_extra_columns_are_ignored(self, tmp_path: Path) -> None:
        # 多出来的列（比如自己加的记号列）不影响读取
        path = write_manifest(
            tmp_path / "m.csv", HEADER + ",我的标记", VALID_ROW + ",待复核"
        )

        assert len(load_manifest(path)) == 1

    def test_column_order_does_not_matter(self, tmp_path: Path) -> None:
        # 按列名取值，不依赖列的位置
        path = write_manifest(
            tmp_path / "m.csv",
            "title,file_name,platform,country,doc_type,publisher,source_url,effective_date",
            "标题,ioss.html,amazon,EU,guideline,eu_commission,https://europa.eu/x,2021-07-01",
        )

        rows = load_manifest(path)

        assert rows[0].file_name == "ioss.html"
        assert rows[0].title == "标题"
        assert rows[0].country == "EU"

    def test_notes_column_may_be_absent(self, tmp_path: Path) -> None:
        # notes 标注为「导入工具不读」，因此不强制存在
        columns = [c for c in MANIFEST_COLUMNS if c != "notes"]
        path = write_manifest(
            tmp_path / "m.csv",
            ",".join(columns),
            "ioss.html,标题,https://europa.eu/x,eu_commission,EU,guideline,2021-07-01,amazon",
        )

        assert load_manifest(path)[0].notes is None


class TestFileLevelErrors:
    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="不存在"):
            load_manifest(tmp_path / "not-there.csv")

    def test_empty_file_is_reported(self, tmp_path: Path) -> None:
        path = tmp_path / "m.csv"
        path.write_text("", encoding="utf-8-sig")

        with pytest.raises(ManifestError, match="表头"):
            load_manifest(path)

    @pytest.mark.parametrize("missing", ["file_name", "title", "country", "platform"])
    def test_missing_column_is_reported(self, tmp_path: Path, missing: str) -> None:
        columns = [c for c in MANIFEST_COLUMNS if c != missing]
        path = write_manifest(tmp_path / "m.csv", ",".join(columns), "x" * len(columns))

        with pytest.raises(ManifestError, match=missing):
            load_manifest(path)


class TestRowLevelErrors:
    def test_duplicate_file_name_is_reported(self, tmp_path: Path) -> None:
        path = manifest_with(tmp_path, VALID_ROW, VALID_ROW)

        with pytest.raises(ManifestError, match="重复"):
            load_manifest(path)

    def test_duplicate_error_names_both_lines(self, tmp_path: Path) -> None:
        path = manifest_with(tmp_path, VALID_ROW, VALID_ROW)

        with pytest.raises(ManifestError) as excinfo:
            load_manifest(path)

        message = str(excinfo.value)
        assert "2" in message and "3" in message

    def test_path_separator_in_file_name_is_rejected(self, tmp_path: Path) -> None:
        # file_name 只应是文件名，不能带路径
        path = manifest_with(
            tmp_path, row(file_name="sub/ioss.html", title="标题")
        )

        with pytest.raises(ManifestError, match="路径"):
            load_manifest(path)

    def test_empty_file_name_is_rejected(self, tmp_path: Path) -> None:
        path = manifest_with(tmp_path, row(title="标题"))

        with pytest.raises(ManifestError, match="file_name"):
            load_manifest(path)

    def test_empty_title_is_rejected(self, tmp_path: Path) -> None:
        path = manifest_with(tmp_path, row(file_name="a.html"))

        with pytest.raises(ManifestError, match="title"):
            load_manifest(path)

    def test_error_message_carries_line_number(self, tmp_path: Path) -> None:
        path = manifest_with(tmp_path, VALID_ROW, row(file_name="a.html"))

        with pytest.raises(ManifestError) as excinfo:
            load_manifest(path)

        assert "第 3 行" in str(excinfo.value)


class TestValueWhitelist:
    @pytest.mark.parametrize(
        "column_index,value",
        [(4, "XX"), (5, "unknown_type"), (3, "some_blog"), (7, "taobao")],
    )
    def test_value_outside_whitelist_is_rejected(
        self, tmp_path: Path, column_index: int, value: str
    ) -> None:
        cells = VALID_ROW.split(",")
        cells[column_index] = value
        path = manifest_with(tmp_path, ",".join(cells))

        with pytest.raises(ManifestError):
            load_manifest(path)

    def test_error_lists_allowed_values(self, tmp_path: Path) -> None:
        # 只说「取值非法」没用，要告诉用户能填什么
        cells = VALID_ROW.split(",")
        cells[4] = "XX"
        path = manifest_with(tmp_path, ",".join(cells))

        with pytest.raises(ManifestError) as excinfo:
            load_manifest(path)

        message = str(excinfo.value)
        assert "EU" in message
        assert "DE" in message

    def test_whitelist_check_is_case_sensitive(self, tmp_path: Path) -> None:
        # "eu" 与 "EU" 混用会让筛选失效，必须统一
        cells = VALID_ROW.split(",")
        cells[4] = "eu"
        path = manifest_with(tmp_path, ",".join(cells))

        with pytest.raises(ManifestError):
            load_manifest(path)


class TestEffectiveDate:
    def test_valid_date_is_parsed(self, tmp_path: Path) -> None:
        rows = load_manifest(manifest_with(tmp_path, VALID_ROW))

        assert rows[0].effective_date == date(2021, 7, 1)

    def test_empty_date_is_allowed(self, tmp_path: Path) -> None:
        # 很多欧盟指南不标生效日期，留空是常态
        rows = load_manifest(
            manifest_with(tmp_path, row(file_name="a.html", title="标题"))
        )

        assert rows[0].effective_date is None

    @pytest.mark.parametrize("bad", ["2021/07/01", "07-01-2021", "去年", "2021-13-01"])
    def test_malformed_date_is_rejected(self, tmp_path: Path, bad: str) -> None:
        cells = VALID_ROW.split(",")
        cells[6] = bad
        path = manifest_with(tmp_path, ",".join(cells))

        with pytest.raises(ManifestError, match="YYYY-MM-DD"):
            load_manifest(path)

    def test_unpadded_date_is_accepted(self, tmp_path: Path) -> None:
        # 手工填写时 "2021-7-1" 很常见，语义无歧义，不必为难使用者
        cells = VALID_ROW.split(",")
        cells[6] = "2021-7-1"
        path = manifest_with(tmp_path, ",".join(cells))

        assert load_manifest(path)[0].effective_date == date(2021, 7, 1)
