"""Redis 适配器测试共用的假对象与构造辅助。

单独放一个模块而不是留在某个测试文件里：适配器本身的测试与问答用例的
测试都要用。
"""

from __future__ import annotations

from typing import Any

from pydantic import SecretStr

from cbe_rag.config.settings import RedisConfig
from cbe_rag.storage.redis_store import RedisStore


def make_config(**overrides: Any) -> RedisConfig:
    fields: dict[str, Any] = {
        "host": "192.168.88.101",
        "port": 6379,
        "password": SecretStr("fake-password"),
        "key_prefix": "cbe",
    }
    fields.update(overrides)
    return RedisConfig(**fields)


class FakeRedis:
    """假的 redis 客户端。

    用内存里的 dict 与 list 模拟存储，够验证键名、TTL 与裁剪行为。
    """

    def __init__(
        self,
        *,
        fail_with: Exception | None = None,
        version: str = "7.2.4",
        info_missing_version: bool = False,
    ) -> None:
        self._fail_with = fail_with
        self._version = version
        self._info_missing_version = info_missing_version
        self.ping_called = 0
        self.closed = False
        self.strings: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.expirations: dict[str, int] = {}

    def _check(self) -> None:
        if self._fail_with is not None:
            raise self._fail_with

    def ping(self) -> bool:
        self.ping_called += 1
        self._check()
        return True

    def info(self, section: str | None = None) -> dict[str, Any]:
        self._check()
        if self._info_missing_version:
            return {}
        return {"redis_version": self._version}

    def rpush(self, name: str, *values: str) -> int:
        self._check()
        bucket = self.lists.setdefault(name, [])
        bucket.extend(values)
        return len(bucket)

    def ltrim(self, name: str, start: int, end: int) -> bool:
        self._check()
        bucket = self.lists.get(name)
        if bucket is None:
            return True
        if start < 0:
            start = max(0, len(bucket) + start)
        if end < 0:
            end = len(bucket) + end
        self.lists[name] = bucket[start : end + 1]
        return True

    def lrange(self, name: str, start: int, end: int) -> list[str]:
        self._check()
        bucket = self.lists.get(name, [])
        if end == -1:
            end = len(bucket) - 1
        return list(bucket[start : end + 1])

    def expire(self, name: str, seconds: int) -> bool:
        self._check()
        self.expirations[name] = seconds
        return True

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        self._check()
        self.strings[name] = value
        if ex is not None:
            self.expirations[name] = ex
        return True

    def get(self, name: str) -> str | None:
        self._check()
        return self.strings.get(name)

    def close(self) -> None:
        self.closed = True


def build_store(
    *, max_messages: int = 20, fail_with: Exception | None = None
) -> tuple[RedisStore, FakeRedis]:
    """构造注入了假客户端的适配器。"""
    config = make_config().model_copy(
        update={"session_max_messages": max_messages}
    )
    client = FakeRedis(fail_with=fail_with)
    return RedisStore(config, client=client), client
