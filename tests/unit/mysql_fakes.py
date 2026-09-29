"""MySQL 适配器测试共用的假对象与构造辅助。

单独放一个模块而不是留在某个测试文件里：两个测试文件都要用，
从一个测试文件 import 另一个会让「哪个是测试、哪个是辅助」变含糊。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import SecretStr

from cbe_rag.config.settings import MysqlConfig
from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.mysql_store import _RECORD_COLUMNS
from cbe_rag.storage.records import DocumentRecord

DEFAULT_ROW = ("8.0.36", "cbe_compliance")


def make_config() -> MysqlConfig:
    """造一份 MySQL 连接配置。"""
    return MysqlConfig(
        host="127.0.0.1",
        port=3306,
        user="cbe",
        password=SecretStr("fake-password"),
        database="cbe_compliance",
    )


class FakeCursor:
    """假的游标。记录执行过的语句与参数，返回预设的行。"""

    def __init__(
        self,
        row: tuple[Any, ...] | None,
        fail_with: Exception | None,
        tables: tuple[str, ...] = (),
        rowcount: int = 1,
        rows: tuple[tuple[Any, ...], ...] = (),
    ) -> None:
        self._row = row
        self._fail_with = fail_with
        self._tables = tables
        self.rowcount = rowcount
        # 默认按 SHOW TABLES 的形状（每行一列）造；一般查询用 rows 直接给
        self._rows = rows or tuple((name,) for name in tables)
        self.executed: list[str] = []
        self.executed_args: list[tuple[Any, ...] | None] = []
        self.executed_many: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.closed = False

    def execute(self, sql: str, args: tuple[Any, ...] | None = None) -> None:
        if self._fail_with is not None:
            raise self._fail_with
        self.executed.append(sql)
        self.executed_args.append(args)

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if self._fail_with is not None:
            raise self._fail_with
        self.executed_many.append((sql, rows))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def fetchall(self) -> tuple[tuple[Any, ...], ...]:
        """返回预设的多行结果。"""
        return self._rows

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """假的连接。记录关闭与事务调用，便于验证连接不泄漏、出错时回滚。"""

    def __init__(
        self,
        row: tuple[Any, ...] | None,
        fail_with: Exception | None,
        tables: tuple[str, ...] = (),
        rowcount: int = 1,
        rows: tuple[tuple[Any, ...], ...] = (),
    ) -> None:
        self.cursor_obj = FakeCursor(row, fail_with, tables, rowcount, rows)
        self.closed = False
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> FakeCursor:
        return self.cursor_obj

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


class FakeConnector:
    """假的连接工厂。

    连接失败（connect_fail_with）与查询失败（query_fail_with）
    是两种不同的故障，分开注入。

    创建过的连接都记录下来，测试通过 .last 取最近一个做断言。
    """

    def __init__(
        self,
        *,
        row: tuple[Any, ...] | None = DEFAULT_ROW,
        query_fail_with: Exception | None = None,
        connect_fail_with: Exception | None = None,
        tables: tuple[str, ...] = (),
        rowcount: int = 1,
        rows: tuple[tuple[Any, ...], ...] = (),
    ) -> None:
        self._row = row
        self._query_fail_with = query_fail_with
        self._connect_fail_with = connect_fail_with
        self._tables = tables
        self._rowcount = rowcount
        self._rows = rows
        self.connections: list[FakeConnection] = []

    def __call__(self) -> FakeConnection:
        if self._connect_fail_with is not None:
            raise self._connect_fail_with
        conn = FakeConnection(
            self._row,
            self._query_fail_with,
            self._tables,
            self._rowcount,
            self._rows,
        )
        self.connections.append(conn)
        return conn

    @property
    def last(self) -> FakeConnection:
        assert self.connections, "尚未创建任何连接"
        return self.connections[-1]


_RECORD_VALUES: dict[str, Any] = {
    "doc_id": "doc-1",
    "content_hash": "a" * 64,
    "status": "indexed",
    "title": "欧洲增值税常见问题",
    "platform": "amazon",
    "source_url": "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX",
    "publisher": "amazon",
    "country": "EU",
    "doc_type": "faq",
    "effective_date": None,
    "collected_date": date(2026, 9, 20),
    "raw_path": "C:/data/raw/amazon-eu-vat-faq.html",
    "missing_fields": None,
    "parse_attempts": None,
}


def make_row(**overrides: Any) -> tuple[Any, ...]:
    """造一行查询结果，按 _RECORD_COLUMNS 的顺序排列。

    列顺序取自实现而不是手写元组：手写的错位后测试仍会「通过」，
    因为两边错得一样。
    """
    values = {**_RECORD_VALUES, **overrides}
    return tuple(values[name] for name in _RECORD_COLUMNS)


def make_record(**overrides: Any) -> DocumentRecord:
    """造一条待写入的文档记录。"""
    fields: dict[str, Any] = {
        "doc_id": "doc-1",
        "content_hash": "a" * 64,
        "status": DocumentStatus.PENDING,
        "title": "欧洲增值税常见问题",
        "platform": "amazon",
        "source_url": "https://example.org/a",
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "collected_date": date(2026, 9, 20),
        "raw_path": "C:/data/raw/a.html",
        "missing_fields": (),
    }
    fields.update(overrides)
    return DocumentRecord(**fields)


def make_chunk(**overrides: Any) -> Chunk:
    """造一个子块。"""
    fields: dict[str, Any] = {
        "chunk_id": "doc-1_c0000",
        "doc_id": "doc-1",
        "parent_id": "doc-1_p0000",
        "level": ChunkLevel.CHILD,
        "chunk_index": 0,
        "text": "进口一站式服务适用于价值不超过 150 欧元的货物。",
        "token_count": 20,
        "start_offset": 0,
        "end_offset": 25,
    }
    fields.update(overrides)
    return Chunk(**fields)


def inserted_values(connector: FakeConnector) -> dict[str, Any]:
    """把 INSERT 的列名与参数值配成字典，便于按列名断言。

    直接按位置断言会在列顺序调整时一起失效——顺序和值同时错，
    测试仍然是绿的。
    """
    cursor = connector.last.cursor_obj
    sql = cursor.executed[0]
    args = cursor.executed_args[0]
    assert args is not None
    columns_part = sql.split("(", 1)[1].split(")", 1)[0]
    columns = [name.strip() for name in columns_part.split(",")]
    return dict(zip(columns, args))
