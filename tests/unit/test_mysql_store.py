"""MySQL 适配器的单元测试：健康检查、生命周期与建表。

注入假的连接工厂，不连接真实服务。假对象与构造辅助见 mysql_fakes.py。
文档与分块的读写测试在同目录的 test_mysql_documents.py。
"""

from __future__ import annotations

import re

import pytest
from pymysql.err import MySQLError, OperationalError

from cbe_rag.storage.ddl import (
    COUNTRY_SEED,
    DOC_TYPE_SEED,
    MYSQL_TABLES,
    PUBLISHER_SEED,
    column_definitions,
)
from cbe_rag.storage.mysql_store import MysqlStore
from mysql_fakes import FakeConnector, make_config


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


def shown_columns(
    table: str, **comment_overrides: str
) -> tuple[tuple[Any, ...], ...]:
    """按 ddl.py 的定义造出 SHOW FULL COLUMNS 的结果。

    默认造出「库里与代码完全一致」的样子；注释可以按列名覆盖，用来造出
    不一致的场景。

    列序：Field, Type, Collation, Null, Key, Default, Extra, Privileges, Comment
    """
    statement = dict(MYSQL_TABLES)[table]
    rows: list[tuple[Any, ...]] = []
    for column, definition in column_definitions(statement).items():
        match = re.search(r"COMMENT\s+'([^']*)'", definition)
        comment = comment_overrides.get(
            column, match.group(1) if match else ""
        )
        rows.append((column, "unknown", None, "NO", "", None, "", "", comment))
    return tuple(rows)


class TestSyncColumns:
    def test_no_change_when_comments_match(self) -> None:
        # 注释一致就不该发 ALTER——每次初始化都重建一遍表结构没有必要
        connector = FakeConnector(rows=shown_columns("documents"))
        store = MysqlStore(make_config(), connect=connector)

        changes = store.sync_columns(tables=("documents",))

        assert changes == []
        assert all(
            "ALTER" not in sql.upper() for sql in connector.last.cursor_obj.executed
        )

    def test_updates_the_comment_when_it_differs(self) -> None:
        # 实测踩过：建表用 IF NOT EXISTS，表已存在时改注释不生效，
        # 库里那列停在旧取值上，看表结构的人被误导
        connector = FakeConnector(rows=shown_columns("documents", doc_id="旧注释"))
        store = MysqlStore(make_config(), connect=connector)

        changes = store.sync_columns(tables=("documents",))

        assert changes == ["documents.doc_id"]
        assert any(
            "ALTER TABLE" in sql.upper() for sql in connector.last.cursor_obj.executed
        )

    def test_alter_carries_the_definition_from_ddl(self) -> None:
        # 用 ddl.py 里的完整定义，类型与可空性一并对齐
        connector = FakeConnector(rows=shown_columns("documents", doc_id="旧注释"))
        store = MysqlStore(make_config(), connect=connector)

        store.sync_columns(tables=("documents",))
        statement = [
            sql for sql in connector.last.cursor_obj.executed if "ALTER" in sql.upper()
        ][0]

        # 列名要带上——column_definitions 给的定义里不含它，漏了就是语法错误。
        # 这个只在真跑时才暴露，所以断言写死一点。
        assert "`doc_id`" in statement
        assert "CHAR(36)" in statement
        assert "NOT NULL" in statement
        assert "文档唯一标识" in statement

    def test_rolls_back_on_failure(self) -> None:
        connector = FakeConnector(
            rows=shown_columns("documents", doc_id="旧注释"),
            query_fail_with=MySQLError("语法错误"),
        )
        store = MysqlStore(make_config(), connect=connector)

        with pytest.raises(MySQLError):
            store.sync_columns(tables=("documents",))

        assert connector.last.rollbacks == 1

    def test_closes_resources(self) -> None:
        connector = FakeConnector(rows=())
        store = MysqlStore(make_config(), connect=connector)

        store.sync_columns(tables=("documents",))

        assert connector.last.closed is True


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
