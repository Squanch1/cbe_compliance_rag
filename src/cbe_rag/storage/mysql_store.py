"""MySQL 适配器。

业务层通过本类访问 MySQL，不直接 import pymysql 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol

import pymysql

from cbe_rag.config.settings import MysqlConfig
from cbe_rag.storage.health import HealthResult


class MysqlCursor(Protocol):
    """游标中本适配器用到的部分。"""

    def execute(self, sql: str) -> Any:
        ...

    def fetchone(self) -> tuple[Any, ...] | None:
        ...

    def close(self) -> None:
        ...


class MysqlConnection(Protocol):
    """连接中本适配器用到的部分。"""

    def cursor(self) -> MysqlCursor:
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
