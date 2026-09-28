"""Redis 适配器。

业务层通过本类访问 Redis，不直接 import redis 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。
"""

from __future__ import annotations

import time
from typing import Any, Protocol

from redis import Redis

from cbe_rag.config.settings import RedisConfig
from cbe_rag.storage.health import HealthResult


class RedisClient(Protocol):
    """redis-py 客户端中本适配器用到的部分。

    单独声明是为了让单元测试能注入假实现，不必连接真实服务。
    """

    def ping(self) -> bool:
        """探测连通性。成功返回 True。"""
        ...

    def info(self, section: str | None = ...) -> dict[str, Any]:
        """返回服务信息。section 为 "server" 时含版本号。"""
        ...

    def close(self) -> None:
        """释放连接池。"""
        ...


class RedisStore:
    """Redis 访问入口。

    构造本对象不会产生网络调用，连接在首次实际操作时建立。
    """

    def __init__(self, config: RedisConfig, client: RedisClient | None = None) -> None:
        """初始化。

        client 仅用于测试注入；生产路径下由配置构造真实客户端。
        """
        self._config = config
        self._client: RedisClient = (
            client
            if client is not None
            else Redis(
                host=config.host,
                port=config.port,
                password=config.password.get_secret_value(),
                socket_timeout=config.timeout_seconds,
                socket_connect_timeout=config.timeout_seconds,
                decode_responses=True,
            )
        )

    def health_check(self) -> HealthResult:
        """探测 Redis 连通性。

        失败时返回 ok=False 的结果而不是抛异常——
        健康检查的职责是报告问题，由调用方决定如何处理。
        """
        started = time.perf_counter()
        try:
            self._client.ping()
            info = self._client.info("server")
            detail = "redis_version=%s" % info.get("redis_version", "unknown")
            ok = True
        except Exception as exc:
            # 刻意捕获所有异常，理由同 MysqlStore.health_check：
            # 失败来源不止驱动异常，漏掉一种就变成调用方崩溃。
            detail = "%s: %s" % (type(exc).__name__, exc)
            ok = False

        return HealthResult(
            service="redis",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放客户端及其连接池。

        redis-py 内部维护连接池，客户端应当长期复用，
        因此不像 MySQL 适配器那样按次建立连接。
        """
        self._client.close()
