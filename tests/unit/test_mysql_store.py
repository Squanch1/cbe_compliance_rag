"""MySQL 适配器的单元测试。

注入假的连接工厂，不连接真实服务。
"""

from __future__ import annotations

from typing import Any

import pytest
from pymysql.err import MySQLError, OperationalError
from pydantic import SecretStr

from cbe_rag.config.settings import MysqlConfig
from cbe_rag.storage.ddl import (
    COUNTRY_SEED,
    DOC_TYPE_SEED,
    MYSQL_TABLES,
    PUBLISHER_SEED,
)
from cbe_rag.storage.mysql_store import MysqlStore

DEFAULT_ROW = ("8.0.36", "cbe_compliance")


def make_config() -> MysqlConfig:
    return MysqlConfig(
        host="127.0.0.1",
        port=3306,
        user="cbe",
        password=SecretStr("fake-password"),
        database="cbe_compliance",
    )


class FakeCursor:
    """假的游标。记录执行过的 SQL，返回预设的行与表名。"""

    def __init__(
        self,
        row: tuple[Any, ...] | None,
        fail_with: Exception | None,
        tables: tuple[str, ...] = (),
    ) -> None:
        self._row = row
        self._fail_with = fail_with
        self._tables = tables
        self.executed: list[str] = []
        self.executed_many: list[tuple[str, list[tuple[Any, ...]]]] = []
        self.closed = False

    def execute(self, sql: str) -> None:
        if self._fail_with is not None:
            raise self._fail_with
        self.executed.append(sql)

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if self._fail_with is not None:
            raise self._fail_with
        self.executed_many.append((sql, rows))

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row

    def fetchall(self) -> tuple[tuple[Any, ...], ...]:
        """模拟 SHOW TABLES 的返回：每行一列。"""
        return tuple((name,) for name in self._tables)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    """假的连接。记录关闭与事务调用，便于验证连接不泄漏、出错时回滚。"""

    def __init__(
        self,
        row: tuple[Any, ...] | None,
        fail_with: Exception | None,
        tables: tuple[str, ...] = (),
    ) -> None:
        self.cursor_obj = FakeCursor(row, fail_with, tables)
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
    ) -> None:
        self._row = row
        self._query_fail_with = query_fail_with
        self._connect_fail_with = connect_fail_with
        self._tables = tables
        self.connections: list[FakeConnection] = []

    def __call__(self) -> FakeConnection:
        if self._connect_fail_with is not None:
            raise self._connect_fail_with
        conn = FakeConnection(self._row, self._query_fail_with, self._tables)
        self.connections.append(conn)
        return conn

    @property
    def last(self) -> FakeConnection:
        assert self.connections, "尚未创建任何连接"
        return self.connections[-1]


class TestHealthCheckSuccess:
    def test_reports_ok_with_version_and_database(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector())

        result = store.health_check()

        assert result.ok is True
        assert result.service == "mysql"
        assert "8.0.36" in result.detail
        assert "cbe_compliance" in result.detail

    def test_queries_version_and_database_in_one_round_trip(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.health_check()

        executed = connector.last.cursor_obj.executed
        assert len(executed) == 1
        assert "VERSION()" in executed[0].upper()
        assert "DATABASE()" in executed[0].upper()

    def test_connection_is_closed_after_check(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.health_check()

        assert connector.last.closed is True

    def test_cursor_is_closed_after_check(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.health_check()

        assert connector.last.cursor_obj.closed is True

    def test_elapsed_time_is_recorded(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector())

        result = store.health_check()

        assert result.elapsed_ms >= 0.0


class TestLifecycle:
    def test_close_is_safe_to_call(self) -> None:
        # 本适配器不持有长期资源，close 是空操作。
        # 方法存在是为了满足 StorageAdapter 协议，让调用方统一遍历。
        store = MysqlStore(make_config(), connect=FakeConnector())

        store.close()

    def test_close_does_not_affect_later_checks(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.close()

        assert store.health_check().ok is True


class TestHealthCheckFailure:
    def test_connection_failure_is_reported_not_raised(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(connect_fail_with=OperationalError("Access denied")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "Access denied" in result.detail

    def test_query_failure_is_reported_not_raised(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(query_fail_with=MySQLError("Unknown database")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "MySQLError" in result.detail

    def test_network_failure_is_reported_not_raised(self) -> None:
        store = MysqlStore(
            make_config(),
            connect=FakeConnector(connect_fail_with=OSError("timed out")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "timed out" in result.detail

    def test_connection_closed_even_when_query_fails(self) -> None:
        # 连接泄漏比一次探测失败更麻烦，异常路径也必须关闭
        connector = FakeConnector(query_fail_with=OperationalError("Access denied"))
        store = MysqlStore(make_config(), connect=connector)

        store.health_check()

        assert connector.last.closed is True

    def test_empty_result_set_is_reported_as_failure(self) -> None:
        # 查询执行了却没返回行，说明协议层有问题，不能当成通过
        store = MysqlStore(make_config(), connect=FakeConnector(row=None))

        result = store.health_check()

        assert result.ok is False

    def test_non_ascii_password_is_reported_not_raised(self) -> None:
        # 回归：pymysql 用 latin-1 编码密码，密码含非 ASCII 字符时
        # 会抛 UnicodeEncodeError。它既不是 MySQLError 也不是 OSError，
        # 早期实现遗漏了这一支，导致 health_check 直接崩溃。
        exc = UnicodeEncodeError("latin-1", "密码", 1, 3, "bad")
        store = MysqlStore(make_config(), connect=FakeConnector(connect_fail_with=exc))

        result = store.health_check()

        assert result.ok is False
        assert "UnicodeEncodeError" in result.detail

    def test_password_is_not_leaked_in_detail(self) -> None:
        config = make_config()
        store = MysqlStore(
            config,
            connect=FakeConnector(connect_fail_with=OperationalError("Access denied")),
        )

        result = store.health_check()

        assert config.password.get_secret_value() not in result.detail


class TestCreateSchema:
    def test_creates_all_tables_when_none_exist(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector(tables=()))

        created = store.create_schema()

        assert set(created) == {name for name, _ in MYSQL_TABLES}

    def test_reports_only_newly_created_tables(self) -> None:
        # 返回值要能区分「这次真的建了」和「本来就有」，
        # 否则脚本输出看着像每次都重建了一遍
        already = MYSQL_TABLES[0][0]
        store = MysqlStore(make_config(), connect=FakeConnector(tables=(already,)))

        created = store.create_schema()

        assert already not in created
        assert len(created) == len(MYSQL_TABLES) - 1

    def test_checks_existing_tables_first(self) -> None:
        connector = FakeConnector(tables=())
        store = MysqlStore(make_config(), connect=connector)

        store.create_schema()

        assert "SHOW TABLES" in connector.last.cursor_obj.executed[0].upper()

    def test_commits_on_success(self) -> None:
        connector = FakeConnector(tables=())
        store = MysqlStore(make_config(), connect=connector)

        store.create_schema()

        assert connector.last.commits == 1

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(tables=(), query_fail_with=MySQLError("语法错误"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.create_schema()

        assert connector.last.rollbacks == 1

    def test_connection_is_closed_even_on_failure(self) -> None:
        connector = FakeConnector(tables=(), query_fail_with=MySQLError("语法错误"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.create_schema()

        assert connector.last.closed is True

    def test_every_ddl_uses_if_not_exists(self) -> None:
        # 缺了这个就没法重复执行
        for name, statement in MYSQL_TABLES:
            assert "IF NOT EXISTS" in statement.upper(), name


class TestSeedDimensions:
    def test_returns_row_count_per_table(self) -> None:
        store = MysqlStore(make_config(), connect=FakeConnector())

        affected = store.seed_dimensions()

        assert affected == {
            "dim_country": len(COUNTRY_SEED),
            "dim_doc_type": len(DOC_TYPE_SEED),
            "dim_publisher": len(PUBLISHER_SEED),
        }

    def test_uses_upsert_not_plain_insert(self) -> None:
        # 纯 INSERT 第二次执行就会撞主键，脚本没法重复跑
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.seed_dimensions()

        for sql, _ in connector.last.cursor_obj.executed_many:
            assert "ON DUPLICATE KEY UPDATE" in sql.upper()

    def test_rows_match_seed_data(self) -> None:
        connector = FakeConnector()
        store = MysqlStore(make_config(), connect=connector)

        store.seed_dimensions()

        counts = [len(rows) for _, rows in connector.last.cursor_obj.executed_many]
        assert counts == [len(COUNTRY_SEED), len(DOC_TYPE_SEED), len(PUBLISHER_SEED)]

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(query_fail_with=MySQLError("表不存在"))
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.seed_dimensions()

        assert connector.last.rollbacks == 1

    def test_seed_data_has_no_duplicate_codes(self) -> None:
        # 主键重复会让 upsert 互相覆盖，最后只剩一条
        for seed in (COUNTRY_SEED, DOC_TYPE_SEED, PUBLISHER_SEED):
            codes = [row[0] for row in seed]
            assert len(codes) == len(set(codes))
