"""文档与分块读写的单元测试。

不连真实服务：假连接见 mysql_fakes.py。这里关心的是「发出去的语句
对不对、参数有没有按位置对上、异常路径有没有回滚」。

按列名断言而不是按位置：列顺序调整时位置断言会跟着一起失效——
顺序和值同时错，测试仍然是绿的。
"""

from __future__ import annotations

import json
from datetime import date, datetime

import pytest
from pymysql.err import MySQLError

from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.mysql_store import _RECORD_COLUMNS, MysqlStore
from mysql_fakes import (
    _RECORD_VALUES,
    FakeConnector,
    inserted_values,
    make_chunk,
    make_config,
    make_record,
    make_row,
)


class TestGetDocumentByHash:
    def test_returns_record_on_hit(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector(row=make_row()))

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.doc_id == "doc-1"
        assert record.status is DocumentStatus.INDEXED

    def test_returns_none_on_miss(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector(row=None))

        assert store.get_document_by_hash("a" * 64) is None

    def test_passes_hash_as_parameter(self) -> None:
        # 哈希不能拼进 SQL：既是注入面，也让驱动没法复用预编译语句
        connector = FakeConnector(row=None)
        store = MysqlStore(make_config(), connect=connector)

        store.get_document_by_hash("deadbeef" * 8)

        cursor = connector.last.cursor_obj
        assert cursor.executed_args == [("deadbeef" * 8,)]
        assert "deadbeef" not in cursor.executed[0]

    def test_queries_by_content_hash(self) -> None:
        connector = FakeConnector(row=None)
        store = MysqlStore(make_config(), connect=connector)

        store.get_document_by_hash("a" * 64)

        assert "content_hash" in connector.last.cursor_obj.executed[0]

    def test_does_not_select_star(self) -> None:
        # SELECT * 在表加字段后列顺序会变，解包随即错位且不报错
        connector = FakeConnector(row=None)
        store = MysqlStore(make_config(), connect=connector)

        store.get_document_by_hash("a" * 64)

        assert "*" not in connector.last.cursor_obj.executed[0]

    def test_closes_cursor_and_connection(self) -> None:
        connector = FakeConnector(row=None)
        store = MysqlStore(make_config(), connect=connector)

        store.get_document_by_hash("a" * 64)

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True

    def test_closes_resources_even_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("连接断了"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.get_document_by_hash("a" * 64)

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True


class TestFindActiveBySourceUrl:
    def test_returns_record_on_hit(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector(row=make_row()))

        record = store.find_active_by_source_url("https://example.org/a")

        assert record is not None
        assert record.source_url == _RECORD_VALUES["source_url"]

    def test_returns_none_on_miss(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector(row=None))

        assert store.find_active_by_source_url("https://example.org/a") is None

    def test_filters_by_indexed_status(self) -> None:
        # 已下线的记录不必查：它们本就不参与检索，再下线一次没有意义
        connector = FakeConnector(row=None)
        store = MysqlStore(make_config(), connect=connector)

        store.find_active_by_source_url("https://example.org/a")

        cursor = connector.last.cursor_obj
        assert "status" in cursor.executed[0]
        assert cursor.executed_args == [
            ("https://example.org/a", DocumentStatus.INDEXED.value)
        ]


class TestRowToRecord:
    def test_parses_missing_fields_json(self) -> None:
        # JSON 列 pymysql 不替我们解析，拿到的是文本
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(
                row=make_row(missing_fields='["source_url", "publisher"]')
            ),
        )

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.missing_fields == ("source_url", "publisher")

    def test_null_missing_fields_becomes_empty_tuple(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(row=make_row(missing_fields=None)),
        )

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.missing_fields == ()

    def test_maps_status_string_to_enum(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(row=make_row(status="superseded")),
        )

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.status is DocumentStatus.SUPERSEDED

    def test_maps_every_column(self) -> None:
        # 列名与 _RECORD_COLUMNS 对不上时会 KeyError，但只在真连库时
        # 才暴露。这里确认造出来的行能被完整解开。
        store = MysqlStore(make_config(), connect=FakeConnector(row=make_row()))

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.title == _RECORD_VALUES["title"]
        assert record.raw_path == _RECORD_VALUES["raw_path"]
        assert record.collected_date == _RECORD_VALUES["collected_date"]
        assert record.effective_date is None

    def test_unknown_status_raises_instead_of_silently_passing(self) -> None:
        # 库里出现枚举之外的值说明数据被改坏了，不能当成某个默认状态
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(row=make_row(status="不存在的状态")),
        )

        with pytest.raises(ValueError):
            store.get_document_by_hash("a" * 64)


class TestInsertDocument:
    def test_writes_the_key_columns(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record())
        values = inserted_values(connector)

        assert values["doc_id"] == "doc-1"
        assert values["content_hash"] == "a" * 64
        assert values["status"] == "pending"
        assert values["title"] == "欧洲增值税常见问题"
        assert values["collected_date"] == date(2026, 9, 20)

    def test_covers_every_record_column_plus_timestamps(self) -> None:
        # 少写一列会让它落到数据库默认值上：NOT NULL 的列直接报错，
        # 可空的列则是静默丢数据
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record())
        values = inserted_values(connector)

        assert set(_RECORD_COLUMNS) <= set(values)
        assert {"created_at", "updated_at"} <= set(values)

    def test_empty_missing_fields_is_stored_as_null(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record(missing_fields=()))
        values = inserted_values(connector)

        assert values["missing_fields"] is None

    def test_missing_fields_are_stored_as_json(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(
            make_record(missing_fields=("source_url", "publisher"))
        )
        values = inserted_values(connector)

        assert json.loads(values["missing_fields"]) == ["source_url", "publisher"]

    def test_commits(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record())

        assert connector.last.commits == 1

    def test_rolls_back_on_failure(self) -> None:
        # 哈希撞唯一约束时走这条路径
        connector = FakeConnector(query_fail_with=MySQLError("Duplicate entry"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.insert_document(make_record())

        assert connector.last.rollbacks == 1

    def test_closes_resources(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record())

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True

    def test_timestamps_are_injectable(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)
        moment = datetime(2026, 9, 29, 12, 0, 0)

        store.insert_document(make_record(), now=moment)
        values = inserted_values(connector)

        assert values["created_at"] == moment
        assert values["updated_at"] == moment


class TestUpdateDocumentMeta:
    def test_does_not_touch_status(self) -> None:
        # 元数据变了不代表文档要重新索引，状态由索引流程决定。
        # 写进这条语句里的话，一次错别字修正就可能把文档打回 pending。
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_meta(make_record())
        sql = connector.last.cursor_obj.executed[0]

        assert "status" not in sql

    def test_writes_the_filter_dimensions(self) -> None:
        # 这三个是检索时的过滤维度，改错了会筛不到
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_meta(make_record(publisher="eu_commission"))
        args = connector.last.cursor_obj.executed_args[0]

        assert args is not None
        assert "eu_commission" in args

    def test_targets_one_document(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_meta(make_record(doc_id="doc-7"))
        args = connector.last.cursor_obj.executed_args[0]

        assert args is not None
        assert args[-1] == "doc-7"

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("表不存在"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.update_document_meta(make_record())

        assert connector.last.rollbacks == 1


class TestUpdateDocumentStatus:
    def test_writes_the_status_value(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status("doc-1", DocumentStatus.INDEXED)

        args = connector.last.cursor_obj.executed_args[0]
        assert args is not None
        assert args[0] == "indexed"

    def test_targets_the_given_document(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status("doc-9", DocumentStatus.SUPERSEDED)

        args = connector.last.cursor_obj.executed_args[0]
        assert args is not None
        assert args[-1] == "doc-9"

    def test_writes_only_status_and_timestamp(self) -> None:
        # 只改状态就只写这两列：元数据这次根本没读出来，
        # 顺带写一遍会把它们清成 NULL
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status("doc-1", DocumentStatus.FAILED)
        sql = connector.last.cursor_obj.executed[0]

        assert "title" not in sql
        assert "publisher" not in sql
        assert "missing_fields" not in sql

    def test_commits(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status("doc-1", DocumentStatus.INDEXED)

        assert connector.last.commits == 1


class TestSupersedeSiblings:
    def test_only_touches_indexed_records(self) -> None:
        # 已下线的再下线一次没有意义；没走完的（pending 等）更不该被动，
        # 它们本来就不可检索
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        args = connector.last.cursor_obj.executed_args[0]
        assert args is not None
        assert args[0] == DocumentStatus.SUPERSEDED.value
        assert args[-1] == DocumentStatus.INDEXED.value

    def test_keeps_the_given_document(self) -> None:
        # 刚索引成功的那条不能被自己收掉
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        sql = connector.last.cursor_obj.executed[0]
        args = connector.last.cursor_obj.executed_args[0]
        assert "doc_id !=" in sql
        assert args is not None
        assert "doc-new" in args

    def test_returns_affected_row_count(self) -> None:
        connector = FakeConnector(rowcount=3)
        store = MysqlStore(make_config(), connect=connector)

        assert store.supersede_siblings("https://example.org/a", "doc-new") == 3

    def test_zero_affected_is_not_an_error(self) -> None:
        # 绝大多数导入都只有一条活跃记录，收掉 0 条是正常情况
        connector = FakeConnector(rowcount=0)
        store = MysqlStore(make_config(), connect=connector)

        assert store.supersede_siblings("https://example.org/a", "doc-new") == 0

    def test_commits(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        assert connector.last.commits == 1


class TestListActiveDocuments:
    def test_returns_all_rows_in_order(self) -> None:
        connector = FakeConnector(
            rows=(make_row(doc_id="a"), make_row(doc_id="b"))
        )
        store = MysqlStore(make_config(), connect=connector)

        records = store.list_active_documents()

        assert [record.doc_id for record in records] == ["a", "b"]

    def test_returns_empty_list_when_none(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        assert store.list_active_documents() == []

    def test_filters_by_indexed(self) -> None:
        # 已下线的、没走完的都不算「库里有这份文档」，混进来会被误报成孤儿
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.list_active_documents()

        assert connector.last.cursor_obj.executed_args == [
            (DocumentStatus.INDEXED.value,)
        ]

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.list_active_documents()

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True


class TestReplaceChunks:
    def test_deletes_before_inserting(self) -> None:
        # 顺序反了会撞 chunk_id 主键
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-1", [make_chunk()])
        cursor = connector.last.cursor_obj

        assert "DELETE" in cursor.executed[0].upper()
        assert len(cursor.executed_many) == 1

    def test_deletes_by_doc_id(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-7", [make_chunk()])

        assert connector.last.cursor_obj.executed_args[0] == ("doc-7",)

    def test_writes_every_chunk_column(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-1", [make_chunk()])
        sql, rows = connector.last.cursor_obj.executed_many[0]
        columns_part = sql.split("(", 1)[1].split(")", 1)[0]
        columns = [name.strip() for name in columns_part.split(",")]
        values = dict(zip(columns, rows[0]))

        assert values["chunk_id"] == "doc-1_c0000"
        assert values["parent_id"] == "doc-1_p0000"
        assert values["level"] == "child"
        assert values["start_offset"] == 0
        assert values["end_offset"] == 25
        assert values["token_count"] == 20

    def test_writes_one_row_per_chunk(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks(
            "doc-1", [make_chunk(chunk_id="a"), make_chunk(chunk_id="b")]
        )
        _, rows = connector.last.cursor_obj.executed_many[0]

        assert len(rows) == 2

    def test_empty_chunk_list_skips_the_insert(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-1", [])

        assert connector.last.cursor_obj.executed_many == []
        assert connector.last.commits == 1

    def test_commits_once_for_both_statements(self) -> None:
        # 删和插在同一事务里：中途失败时旧的还在，
        # 不会变成「块被删光了」或「一半新一半旧」
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-1", [make_chunk()])

        assert connector.last.commits == 1

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("插入失败"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.replace_chunks("doc-1", [make_chunk()])

        assert connector.last.rollbacks == 1

    def test_closes_resources(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.replace_chunks("doc-1", [make_chunk()])

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True

    def test_timestamp_is_injectable(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)
        moment = datetime(2026, 9, 29, 12, 0, 0)

        store.replace_chunks("doc-1", [make_chunk()], now=moment)
        _, rows = connector.last.cursor_obj.executed_many[0]

        assert rows[0][-1] == moment
