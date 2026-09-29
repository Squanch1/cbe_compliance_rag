"""MySQL 适配器。

业务层通过本类访问 MySQL，不直接 import pymysql 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import pymysql

from cbe_rag.config.settings import MysqlConfig
from cbe_rag.storage.ddl import (
    COUNTRY_SEED,
    DOC_TYPE_SEED,
    MYSQL_TABLES,
    PUBLISHER_SEED,
)
from cbe_rag.storage.health import HealthResult


class MysqlCursor(Protocol):
    """游标中本适配器用到的部分。"""

    def execute(self, sql: str) -> Any:
        ...

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> Any:
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        ...

    def fetchall(self) -> tuple[tuple[Any, ...], ...]:
        ...

    def close(self) -> None:
        ...


class MysqlConnection(Protocol):
    """连接中本适配器用到的部分。"""

    def cursor(self) -> MysqlCursor:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


def _close_quietly(resource: Any) -> None:
    """关闭资源，忽略关闭过程中的异常。

    关闭失败不应覆盖健康检查已经得出的结论。
    """
    if resource is None:
        return
    try:
        resource.close()
    except Exception:
        pass


class MysqlStore:
    """MySQL 访问入口。

    构造本对象不会产生网络调用，连接在每次操作时建立并关闭。
    连接池留到性能确有需要时再引入。
    """

    def __init__(
        self,
        config: MysqlConfig,
        connect: Callable[[], MysqlConnection] | None = None,
    ) -> None:
        """初始化。

        connect 仅用于测试注入；生产路径下由配置构造真实连接。
        """
        self._config = config
        self._connect: Callable[[], MysqlConnection] = (
            connect if connect is not None else self._connect_by_config
        )

    def _connect_by_config(self) -> MysqlConnection:
        """按配置建立真实连接。"""
        timeout = max(1, int(self._config.timeout_seconds))
        return pymysql.connect(
            host=self._config.host,
            port=self._config.port,
            user=self._config.user,
            password=self._config.password.get_secret_value(),
            database=self._config.database,
            charset=self._config.charset,
            connect_timeout=timeout,
            read_timeout=timeout,
            write_timeout=timeout,
            cursorclass=pymysql.cursors.Cursor,
        )

    def health_check(self) -> HealthResult:
        """探测 MySQL 连通性，并确认连到的是预期的库。

        一次往返同时取版本号与当前库名：连错库是部署时的常见错误，
        只报版本号看不出这个问题。

        失败时返回 ok=False 的结果而不是抛异常。
        """
        started = time.perf_counter()
        connection: MysqlConnection | None = None
        cursor: MysqlCursor | None = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION(), DATABASE()")
            row = cursor.fetchone()
            if row is None:
                ok = False
                detail = "查询执行成功但未返回结果行"
            else:
                ok = True
                detail = "version=%s database=%s" % (row[0], row[1])
        except Exception as exc:
            # 这里刻意捕获所有异常。健康检查的契约是报告问题而非抛异常，
            # 而失败来源不止驱动异常——例如密码含非 ASCII 字符时，
            # pymysql 会在编码阶段抛 UnicodeEncodeError，它既不是
            # MySQLError 也不是 OSError。漏掉一种就变成调用方崩溃。
            ok = False
            detail = "%s: %s" % (type(exc).__name__, exc)
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

        return HealthResult(
            service="mysql",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """空操作。

        本适配器每次操作自行建立并关闭连接，不持有需要释放的资源。
        方法存在是为了满足 StorageAdapter 协议，让调用方能统一遍历所有适配器。
        """
        return None

    def create_schema(self) -> list[str]:
        """建表，返回本次新建的表名。

        DDL 全部带 IF NOT EXISTS，因此可重复执行。返回的是**新建**的表名
        而不是全部表名，让调用方能区分「这次真的建了」和「本来就有」。
        """
        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()

            # 先查现有表，才能分辨哪些是这次新建的
            cursor.execute("SHOW TABLES")
            existing = {row[0] for row in cursor.fetchall()}

            created: list[str] = []
            for name, statement in MYSQL_TABLES:
                cursor.execute(statement)
                if name not in existing:
                    created.append(name)

            connection.commit()
            return created
        except Exception:
            # 出错时回滚再抛出。这里捕获所有异常只是为了确保回滚，
            # 异常本身照常向上传播。
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)

    def seed_dimensions(self) -> dict[str, int]:
        """灌入三张维度表的初始数据，返回每张表影响的行数。

        用 INSERT ... ON DUPLICATE KEY UPDATE，因此可重复执行。
        **以 ddl.py 里的定义为准**：直接改库里的值会被下次执行覆盖，
        要调整维度数据应当改 ddl.py（见该模块的注释）。
        """
        statements: list[tuple[str, str, list[tuple[Any, ...]]]] = [
            (
                "dim_country",
                "INSERT INTO dim_country (code, name_zh, name_en, is_active) "
                "VALUES (%s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, is_active = new.is_active",
                [tuple(row) for row in COUNTRY_SEED],
            ),
            (
                "dim_doc_type",
                "INSERT INTO dim_doc_type (code, name_zh, name_en, is_active) "
                "VALUES (%s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, is_active = new.is_active",
                [tuple(row) for row in DOC_TYPE_SEED],
            ),
            (
                "dim_publisher",
                "INSERT INTO dim_publisher (code, name_zh, name_en, official_url, is_active) "
                "VALUES (%s, %s, %s, %s, %s) AS new "
                "ON DUPLICATE KEY UPDATE name_zh = new.name_zh, "
                "name_en = new.name_en, official_url = new.official_url, "
                "is_active = new.is_active",
                [tuple(row) for row in PUBLISHER_SEED],
            ),
        ]

        connection = self._connect()
        cursor = None
        try:
            cursor = connection.cursor()
            affected: dict[str, int] = {}
            for name, statement, rows in statements:
                cursor.executemany(statement, rows)
                affected[name] = len(rows)
            connection.commit()
            return affected
        except Exception:
            # 出错时回滚再抛出。这里捕获所有异常只是为了确保回滚，
            # 异常本身照常向上传播。
            connection.rollback()
            raise
        finally:
            _close_quietly(cursor)
            _close_quietly(connection)
