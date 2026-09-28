"""Redis 适配器的单元测试。

注入假客户端，不连接真实服务。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr
from redis.exceptions import AuthenticationError, ConnectionError as RedisConnectionError

from cbe_rag.config.settings import RedisConfig
from cbe_rag.storage.redis_store import RedisStore


def make_config() -> RedisConfig:
    return RedisConfig(
        host="192.168.88.101",
        port=6379,
        password=SecretStr("fake-password"),
        key_prefix="cbe",
    )


class FakeRedis:
    """假的 redis 客户端。

    只实现本适配器用到的两个方法，行为由构造参数控制。
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

    def ping(self) -> bool:
        self.ping_called += 1
        if self._fail_with is not None:
            raise self._fail_with
        return True

    def info(self, section: str | None = None) -> dict[str, Any]:
        if self._fail_with is not None:
            raise self._fail_with
        if self._info_missing_version:
            return {}
        return {"redis_version": self._version}

    def close(self) -> None:
        self.closed = True


class TestHealthCheckSuccess:
    def test_reports_ok_and_version(self) -> None:
        client = FakeRedis(version="7.2.4")
        store = RedisStore(make_config(), client=client)

        result = store.health_check()

        assert result.ok is True
        assert result.service == "redis"
        assert "7.2.4" in result.detail

    def test_ping_is_actually_called(self) -> None:
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.health_check()

        assert client.ping_called == 1

    def test_elapsed_time_is_recorded(self) -> None:
        store = RedisStore(make_config(), client=FakeRedis())

        result = store.health_check()

        assert result.elapsed_ms >= 0.0

    def test_missing_version_field_does_not_crash(self) -> None:
        # 某些托管 Redis 的 info("server") 不返回 redis_version
        store = RedisStore(make_config(), client=FakeRedis(info_missing_version=True))

        result = store.health_check()

        assert result.ok is True
        assert "unknown" in result.detail


class TestHealthCheckFailure:
    @pytest.mark.parametrize(
        "exc",
        [
            AuthenticationError("Authentication required."),
            RedisConnectionError("Connection refused"),
            OSError("timed out"),
        ],
    )
    def test_failure_is_reported_not_raised(self, exc: Exception) -> None:
        # 健康检查的职责是报告问题，不是把异常抛给调用方
        store = RedisStore(make_config(), client=FakeRedis(fail_with=exc))

        result = store.health_check()

        assert result.ok is False

    def test_failure_detail_names_the_exception_type(self) -> None:
        exc = AuthenticationError("Authentication required.")
        store = RedisStore(make_config(), client=FakeRedis(fail_with=exc))

        result = store.health_check()

        assert "AuthenticationError" in result.detail
        assert "Authentication required." in result.detail

    def test_unexpected_exception_is_reported_not_raised(self) -> None:
        # 回归：失败来源不止驱动异常，任何异常都不得穿透 health_check
        store = RedisStore(make_config(), client=FakeRedis(fail_with=ValueError("boom")))

        result = store.health_check()

        assert result.ok is False
        assert "ValueError" in result.detail

    def test_password_is_not_leaked_in_failure_detail(self) -> None:
        config = make_config()
        store = RedisStore(
            config,
            client=FakeRedis(fail_with=RedisConnectionError("Connection refused")),
        )

        result = store.health_check()

        assert config.password.get_secret_value() not in result.detail


class TestLifecycle:
    def test_health_check_does_not_close_the_client(self) -> None:
        # redis-py 内部维护连接池，客户端应当长期复用
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.health_check()

        assert client.closed is False

    def test_close_releases_the_client(self) -> None:
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.close()

        assert client.closed is True


class TestHealthResultRendering:
    def test_ok_line_contains_service_and_timing(self) -> None:
        store = RedisStore(make_config(), client=FakeRedis(version="7.2.4"))

        line = store.health_check().render()

        assert line.startswith("[OK")
        assert "redis" in line
        assert "7.2.4" in line

    def test_fail_line_is_marked(self) -> None:
        store = RedisStore(
            make_config(),
            client=FakeRedis(fail_with=RedisConnectionError("Connection refused")),
        )

        line = store.health_check().render()

        assert line.startswith("[FAIL]")
