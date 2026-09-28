"""MongoDB 适配器。

业务层通过本类访问 MongoDB，不直接 import pymongo 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。

与 MySQL 适配器的差异：pymongo 自带连接池且客户端设计为长期复用，
因此本适配器持有客户端直到显式 close()，不按次建立连接。
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from pymongo import MongoClient

from cbe_rag.config.settings import MongodbConfig
from cbe_rag.storage.health import HealthResult


class MongoDatabase(Protocol):
    """数据库句柄中本适配器用到的部分。"""

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        ...


class MongoClientProtocol(Protocol):
    """pymongo 客户端中本适配器用到的部分。

    单独声明是为了让单元测试能注入假实现，不必连接真实服务。
    """

    @property
    def admin(self) -> MongoDatabase:
        ...

    def server_info(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class MongoStore:
    """MongoDB 访问入口。

    构造本对象不会产生网络调用，pymongo 在首次实际操作时才连接。
    """

    def __init__(
        self,
        config: MongodbConfig,
        client: MongoClientProtocol | None = None,
    ) -> None:
        """初始化。

        client 仅用于测试注入；生产路径下由配置构造真实客户端。
        """
        self._config = config
        self._client: MongoClientProtocol = (
            client if client is not None else self._connect_by_config()
        )

    def _connect_by_config(self) -> MongoClientProtocol:
        """按配置建立真实客户端。"""
        timeout_ms = max(1000, int(self._config.timeout_seconds * 1000))
        return MongoClient(
            host=self._config.host,
            port=self._config.port,
            username=self._config.user,
            password=self._config.password.get_secret_value(),
            authSource=self._config.auth_source,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
        )

    def health_check(self) -> HealthResult:
        """探测 MongoDB 连通性。

        先发 ping 确认可达与认证通过，再取服务端版本号。
        两项分开是因为失败原因不同：ping 失败多为网络或认证问题，
        取版本号失败多为权限不足，排查方向不一样。

        失败时返回 ok=False 的结果而不是抛异常。
        """
        started = time.perf_counter()
        try:
            self._client.admin.command({"ping": 1})
            info = self._client.server_info()
            version = info.get("version", "unknown")
            ok = True
            detail = "version=%s auth_source=%s" % (version, self._config.auth_source)
        except Exception as exc:
            # 刻意捕获所有异常。健康检查的契约是报告问题而非抛异常，
            # 失败来源不止驱动异常，漏掉一种就变成调用方崩溃。
            ok = False
            detail = "%s: %s" % (type(exc).__name__, exc)

        return HealthResult(
            service="mongodb",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放客户端及其连接池。由调用方在不再使用时显式调用。"""
        self._client.close()
