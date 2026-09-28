"""MongoDB 适配器的单元测试。

注入假的客户端，不连接真实服务。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from cbe_rag.config.settings import MongodbConfig
from cbe_rag.storage.mongo_store import MongoStore


def make_config() -> MongodbConfig:
    return MongodbConfig(
        host="192.168.88.101",
        port=27017,
        user="cbe",
        password=SecretStr("fake-password"),
        database="cbe_compliance",
        auth_source="admin",
    )


class FakeDatabase:
    """假的数据库句柄。记录收到的命令。"""

    def __init__(self, fail_with: Exception | None) -> None:
        self._fail_with = fail_with
        self.commands: list[dict[str, Any]] = []

    def command(self, command: dict[str, Any]) -> dict[str, Any]:
        if self._fail_with is not None:
            raise self._fail_with
        self.commands.append(command)
        return {"ok": 1.0}


class FakeMongoClient:
    """假的 pymongo 客户端。"""

    def __init__(
        self,
        *,
        fail_with: Exception | None = None,
        info_fail_with: Exception | None = None,
        version: str = "7.0.5",
        omit_version: bool = False,
    ) -> None:
        self._fail_with = fail_with
        self._info_fail_with = info_fail_with
        self._version = version
        self._omit_version = omit_version
        self._admin = FakeDatabase(fail_with)
        self.closed = False

    @property
    def admin(self) -> FakeDatabase:
        return self._admin

    def server_info(self) -> dict[str, Any]:
        if self._info_fail_with is not None:
            raise self._info_fail_with
        if self._omit_version:
            return {}
        return {"version": self._version}

    def close(self) -> None:
        self.closed = True


class TestHealthCheckSuccess:
    def test_reports_ok_with_version(self) -> None:
        store = MongoStore(make_config(), client=FakeMongoClient(version="7.0.5"))

        result = store.health_check()

        assert result.ok is True
        assert result.service == "mongodb"
        assert "7.0.5" in result.detail

    def test_detail_includes_auth_source(self) -> None:
        # 认证库配错是连不上的常见原因，写进结果便于排查
        store = MongoStore(make_config(), client=FakeMongoClient())

        result = store.health_check()

        assert "admin" in result.detail

    def test_ping_command_is_sent(self) -> None:
        client = FakeMongoClient()
        store = MongoStore(make_config(), client=client)

        store.health_check()

        assert client.admin.commands == [{"ping": 1}]

    def test_elapsed_time_is_recorded(self) -> None:
        store = MongoStore(make_config(), client=FakeMongoClient())

        result = store.health_check()

        assert result.elapsed_ms >= 0.0

    def test_missing_version_field_does_not_crash(self) -> None:
        store = MongoStore(make_config(), client=FakeMongoClient(omit_version=True))

        result = store.health_check()

        assert result.ok is True
        assert "unknown" in result.detail


class TestHealthCheckFailure:
    def test_connection_timeout_is_reported_not_raised(self) -> None:
        store = MongoStore(
            make_config(),
            client=FakeMongoClient(fail_with=ServerSelectionTimeoutError("timed out")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "ServerSelectionTimeoutError" in result.detail

    def test_auth_failure_is_reported_not_raised(self) -> None:
        store = MongoStore(
            make_config(),
            client=FakeMongoClient(fail_with=OperationFailure("Authentication failed")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "Authentication failed" in result.detail

    def test_server_info_failure_is_reported_not_raised(self) -> None:
        # ping 通了但取版本号失败，同样不能放行
        store = MongoStore(
            make_config(),
            client=FakeMongoClient(info_fail_with=OperationFailure("not authorized")),
        )

        result = store.health_check()

        assert result.ok is False
        assert "not authorized" in result.detail

    def test_unexpected_exception_is_reported_not_raised(self) -> None:
        # 回归：失败来源不止驱动异常，任何异常都不得穿透 health_check
        store = MongoStore(make_config(), client=FakeMongoClient(fail_with=ValueError("boom")))

        result = store.health_check()

        assert result.ok is False
        assert "ValueError" in result.detail

    def test_password_is_not_leaked_in_detail(self) -> None:
        config = make_config()
        store = MongoStore(
            config,
            client=FakeMongoClient(fail_with=OperationFailure("Authentication failed")),
        )

        result = store.health_check()

        assert config.password.get_secret_value() not in result.detail


class TestLifecycle:
    def test_health_check_does_not_close_the_client(self) -> None:
        # pymongo 自带连接池，客户端应当长期复用。
        # 每次探测都关掉会破坏连接复用，与 MySQL 适配器的处理不同。
        client = FakeMongoClient()
        store = MongoStore(make_config(), client=client)

        store.health_check()

        assert client.closed is False

    def test_close_releases_the_client(self) -> None:
        client = FakeMongoClient()
        store = MongoStore(make_config(), client=client)

        store.close()

        assert client.closed is True


class TestHealthResultRendering:
    def test_ok_line_contains_service(self) -> None:
        store = MongoStore(make_config(), client=FakeMongoClient(version="7.0.5"))

        line = store.health_check().render()

        assert line.startswith("[OK")
        assert "mongodb" in line

    def test_fail_line_is_marked(self) -> None:
        store = MongoStore(
            make_config(),
            client=FakeMongoClient(fail_with=ServerSelectionTimeoutError("timed out")),
        )

        line = store.health_check().render()

        assert line.startswith("[FAIL]")
