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
from cbe_rag.storage.mysql_store import (
    _CHUNK_COLUMNS,
    _RECORD_COLUMNS,
    MysqlStore,
)
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

    def test_parses_parse_attempts_json(self) -> None:
        # 调阈值时要看分布，而分布只能从这些原始分数里来
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(
                row=make_row(
                    parse_attempts='[{"tier": "pdf.text_layer", "ok": false}]'
                )
            ),
        )

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.parse_attempts[0]["tier"] == "pdf.text_layer"

    def test_null_parse_attempts_becomes_empty_tuple(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(row=make_row(parse_attempts=None)),
        )

        record = store.get_document_by_hash("a" * 64)

        assert record is not None
        assert record.parse_attempts == ()

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

    def test_empty_parse_attempts_is_stored_as_null(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.insert_document(make_record())
        values = inserted_values(connector)

        assert values["parse_attempts"] is None

    def test_parse_attempts_are_stored_as_json(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)
        record = make_record(
            parse_attempts=({"tier": "pdf.text_layer", "ok": False},)
        )

        store.insert_document(record)
        values = inserted_values(connector)

        assert json.loads(values["parse_attempts"])[0]["tier"] == "pdf.text_layer"

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

    def test_writes_parse_attempts_when_given(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status(
            "doc-1",
            DocumentStatus.NEEDS_MANUAL,
            parse_attempts=[{"tier": "pdf.text_layer", "ok": False}],
        )

        sql = connector.last.cursor_obj.executed[0]
        args = connector.last.cursor_obj.executed_args[0]
        assert "parse_attempts" in sql
        assert args is not None
        assert json.loads(args[2])[0]["tier"] == "pdf.text_layer"

    def test_omitting_parse_attempts_leaves_the_column_alone(self) -> None:
        # 不传时不能把它清掉：只改状态是「这次不该碰它」，
        # 和「解析过但没有记录可写」是两回事
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.update_document_status("doc-1", DocumentStatus.SUPERSEDED)
        sql = connector.last.cursor_obj.executed[0]

        assert "parse_attempts" not in sql


class TestSupersedeSiblings:
    def test_returns_the_ids_that_were_taken_down(self) -> None:
        # 调用方要拿这些 id 去删 Milvus 里的向量——只改状态拦不住检索
        connector = FakeConnector(rows=(("doc-old",), ("doc-older",)))
        store = MysqlStore(make_config(), connect=connector)

        taken_down = store.supersede_siblings("https://example.org/a", "doc-new")

        assert taken_down == ["doc-old", "doc-older"]

    def test_selects_before_updating(self) -> None:
        # 顺序反了就拿不到要下线的 id 了
        connector = FakeConnector(rows=(("doc-old",),))
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        executed = connector.last.cursor_obj.executed
        assert executed[0].upper().startswith("SELECT")
        assert executed[1].upper().startswith("UPDATE")

    def test_only_touches_indexed_records(self) -> None:
        # 已下线的再下线一次没有意义；没走完的（pending 等）更不该被动，
        # 它们本来就不可检索
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        args = connector.last.cursor_obj.executed_args[0]
        assert args is not None
        assert args[-1] == DocumentStatus.INDEXED.value

    def test_writes_the_superseded_status(self) -> None:
        connector = FakeConnector(rows=(("doc-old",),))
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        args = connector.last.cursor_obj.executed_args[1]
        assert args is not None
        assert args[0] == DocumentStatus.SUPERSEDED.value

    def test_keeps_the_given_document(self) -> None:
        # 刚索引成功的那条不能被自己收掉
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        sql = connector.last.cursor_obj.executed[0]
        args = connector.last.cursor_obj.executed_args[0]
        assert "doc_id !=" in sql
        assert args is not None
        assert "doc-new" in args

    def test_no_siblings_skips_the_update(self) -> None:
        # 绝大多数导入都只有一条活跃记录，这时不必发那条 UPDATE
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        assert store.supersede_siblings("https://example.org/a", "doc-new") == []
        assert len(connector.last.cursor_obj.executed) == 1

    def test_commits(self) -> None:
        connector = FakeConnector(rows=(("doc-old",),))
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        assert connector.last.commits == 1

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("表不存在"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.supersede_siblings("https://example.org/a", "doc-new")

        assert connector.last.rollbacks == 1

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.supersede_siblings("https://example.org/a", "doc-new")

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True


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


_CHUNK_VALUES: dict[str, Any] = {
    "chunk_id": "doc-1_p0000",
    "doc_id": "doc-1",
    "parent_id": None,
    "level": "parent",
    "chunk_index": 0,
    "text": "进口一站式服务适用于价值不超过 150 欧元的货物。",
    "token_count": 20,
    "start_offset": None,
    "end_offset": None,
}


def make_chunk_row(**overrides: Any) -> tuple[Any, ...]:
    """造一行分块查询结果，按 _CHUNK_COLUMNS 的顺序排列。

    默认造的是父块（没有 parent_id 与偏移）；子块要自己覆盖那三项。
    """
    values = {**_CHUNK_VALUES, **overrides}
    return tuple(values[name] for name in _CHUNK_COLUMNS)


class TestGetChunks:
    def test_returns_chunks(self) -> None:
        connector = FakeConnector(rows=(make_chunk_row(),))
        store = MysqlStore(make_config(), connect=connector)

        chunks = store.get_chunks(["doc-1_p0000"])

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc-1_p0000"
        assert chunks[0].text == _CHUNK_VALUES["text"]

    def test_returns_empty_list_for_empty_input(self) -> None:
        # IN () 是语法错误，空列表必须提前返回，连连接都不该建
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        assert store.get_chunks([]) == []
        assert connector.connections == []

    def test_uses_one_statement_for_the_whole_batch(self) -> None:
        # 逐条查的话，保留 5 个父块就是 5 次往返，而它们本来就是同一批
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_chunks(["a", "b", "c"])

        assert len(connector.last.cursor_obj.executed) == 1

    def test_placeholders_match_the_id_count(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_chunks(["a", "b", "c"])

        cursor = connector.last.cursor_obj
        assert cursor.executed[0].count("%s") == 3
        assert cursor.executed_args[0] == ("a", "b", "c")

    def test_ids_are_bound_not_interpolated(self) -> None:
        # 拼进 SQL 既是注入面，也让驱动没法复用预编译语句
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_chunks(["x' OR '1'='1"])

        assert "OR" not in connector.last.cursor_obj.executed[0]

    def test_parent_chunk_keeps_a_null_parent_id(self) -> None:
        # 父块没有 parent_id，转成 Chunk 时不能变成空字符串
        connector = FakeConnector(rows=(make_chunk_row(),))
        store = MysqlStore(make_config(), connect=connector)

        assert store.get_chunks(["doc-1_p0000"])[0].parent_id is None

    def test_child_chunk_keeps_its_offsets(self) -> None:
        connector = FakeConnector(
            rows=(
                make_chunk_row(
                    chunk_id="doc-1_c0000",
                    parent_id="doc-1_p0000",
                    level="child",
                    start_offset=0,
                    end_offset=25,
                ),
            )
        )
        store = MysqlStore(make_config(), connect=connector)

        chunk = store.get_chunks(["doc-1_c0000"])[0]

        assert chunk.parent_id == "doc-1_p0000"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 25

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_chunks(["a"])

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True

    def test_closes_resources_even_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("表不存在"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.get_chunks(["a"])

        assert connector.last.closed is True


class TestListDimensions:
    def executed_statements(self, connector: FakeConnector) -> list[str]:
        """把所有连接上执行过的语句收齐。

        三张表各查一次、各建一次连接（_fetch_many 一次查询一个连接），
        connector.last 只看得到最后一条。
        """
        return [
            statement
            for connection in connector.connections
            for statement in connection.cursor_obj.executed
        ]

    def test_queries_all_three_tables(self) -> None:
        connector = FakeConnector(rows=(("EU", "欧盟", "European Union"),))
        store = MysqlStore(make_config(), connect=connector)

        store.list_dimensions()

        executed = self.executed_statements(connector)
        assert len(executed) == 3
        for table in ("dim_country", "dim_doc_type", "dim_publisher"):
            assert any(table in statement for statement in executed)

    def test_only_reads_active_values(self) -> None:
        # 未启用的取值是给扩充范围预留的，摆在界面上只会让人选到一个
        # 筛不出东西的值
        connector = FakeConnector(rows=(("EU", "欧盟", "European Union"),))
        store = MysqlStore(make_config(), connect=connector)

        store.list_dimensions()

        for statement in self.executed_statements(connector):
            assert "is_active = 1" in statement

    def test_maps_rows_to_dimensions(self) -> None:
        connector = FakeConnector(rows=(("EU", "欧盟", "European Union"),))
        store = MysqlStore(make_config(), connect=connector)

        dimensions = store.list_dimensions()

        assert dimensions.countries[0].code == "EU"
        assert dimensions.countries[0].name_zh == "欧盟"
        assert dimensions.countries[0].name_en == "European Union"

    def test_empty_tables_yield_empty_lists(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        dimensions = store.list_dimensions()

        assert dimensions.countries == []
        assert dimensions.doc_types == []
        assert dimensions.publishers == []

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.list_dimensions()

        assert connector.last.closed is True


class TestGetDocuments:
    def test_returns_records(self) -> None:
        connector = FakeConnector(rows=(make_row(),))
        store = MysqlStore(make_config(), connect=connector)

        records = store.get_documents(["doc-1"])

        assert len(records) == 1
        assert records[0].doc_id == "doc-1"
        assert records[0].title == _RECORD_VALUES["title"]

    def test_returns_empty_list_for_empty_input(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        assert store.get_documents([]) == []
        assert connector.connections == []

    def test_uses_one_statement_for_the_whole_batch(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_documents(["a", "b"])

        assert len(connector.last.cursor_obj.executed) == 1

    def test_placeholders_match_the_id_count(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_documents(["a", "b"])

        cursor = connector.last.cursor_obj
        assert cursor.executed[0].count("%s") == 2
        assert cursor.executed_args[0] == ("a", "b")

    def test_does_not_select_star(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_documents(["a"])

        assert "*" not in connector.last.cursor_obj.executed[0]

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.get_documents(["a"])

        assert connector.last.cursor_obj.closed is True
        assert connector.last.closed is True
