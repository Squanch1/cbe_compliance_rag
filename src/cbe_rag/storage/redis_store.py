"""Redis 适配器。

业务层通过本类访问 Redis，不直接 import redis 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。
"""

from __future__ import annotations

import json
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

    def rpush(self, name: str, *values: str) -> int:
        """从右侧追加。"""
        ...

    def ltrim(self, name: str, start: int, end: int) -> bool:
        """裁剪列表到指定区间。"""
        ...

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        """读取列表区间。"""
        ...

    def expire(self, name: str, seconds: int) -> bool:
        """设置过期时间。"""
        ...

    def set(self, name: str, value: str, ex: int | None = ...) -> bool:
        """写入字符串。"""
        ...

    def get(self, name: str) -> str | None:
        """读取字符串。"""
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

    def _session_key(self, session_id: str) -> str:
        """会话上下文的键。

        带 key_prefix 是因为这个实例由多个项目共用（见 RedisConfig 的说明），
        不加前缀会和其他项目的键混在一起。
        """
        return "%s:session:%s" % (self._config.key_prefix, session_id)

    def _cache_key(self, digest: str) -> str:
        """问答缓存的键。"""
        return "%s:cache:%s" % (self._config.key_prefix, digest)

    def append_message(self, session_id: str, message: dict[str, Any]) -> None:
        """把一条会话消息追加进上下文，并续期。

        **用列表而不是一串独立的键**：一轮问答是两条消息（问与答），要按
        顺序整体读回来，列表的语义正好，`lrange` 一次就取全。

        超出条数上限时从最早的丢起——多轮对话里最早那几轮对当前问题的
        参考价值最低，而上下文长度直接影响每次调用的成本。
        """
        key = self._session_key(session_id)
        self._client.rpush(key, json.dumps(message, ensure_ascii=False))
        self._client.ltrim(key, -self._config.session_max_messages, -1)
        self._client.expire(key, self._config.session_ttl_seconds)

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """读回一段会话的全部消息，按先后顺序。没有则返回空列表。"""
        raw = self._client.lrange(self._session_key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def cache_answer(self, digest: str, payload: dict[str, Any]) -> None:
        """缓存一次问答结果。

        digest 由调用方算好（问题与过滤条件的哈希），适配器不关心它怎么来
        ——那属于业务规则。
        """
        self._client.set(
            self._cache_key(digest),
            json.dumps(payload, ensure_ascii=False),
            ex=self._config.cache_ttl_seconds,
        )

    def get_cached_answer(self, digest: str) -> dict[str, Any] | None:
        """取缓存的问答结果，没有则返回 None。"""
        raw = self._client.get(self._cache_key(digest))
        return None if raw is None else json.loads(raw)

    def close(self) -> None:
        """释放客户端及其连接池。

        redis-py 内部维护连接池，客户端应当长期复用，
        因此不像 MySQL 适配器那样按次建立连接。
        """
        self._client.close()
