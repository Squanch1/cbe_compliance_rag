"""Milvus 适配器的单元测试。

注入假的客户端工厂，不连接真实服务。
"""

from __future__ import annotations

from pymilvus.exceptions import MilvusException

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.milvus_store import MilvusStore


def make_config() -> MilvusConfig:
    return MilvusConfig(
        host="192.168.88.101",
        port=19530,
        database="cbe_compliance",
        collection="chunks_v1",
    )


class FakeMilvusClient:
    """假的 MilvusClient。

    多种失败分开注入，因为它们对应不同的排查方向。
    """

    def __init__(
        self,
        *,
        fail_with: Exception | None = None,
        list_fail_with: Exception | None = None,
        version: str = "2.6.6",
        databases: list[str] | None = None,
    ) -> None:
        self._fail_with = fail_with
        self._list_fail_with = list_fail_with
        self._version = version
        self._databases = ["default"] if databases is None else databases
        self.closed = False

    def get_server_version(self) -> str:
        if self._fail_with is not None:
            raise self._fail_with
        return self._version

    def list_databases(self) -> list[str]:
        if self._fail_with is not None:
            raise self._fail_with
        if self._list_fail_with is not None:
            raise self._list_fail_with
        return list(self._databases)

    def close(self) -> None:
        self.closed = True


def connect_error(message: str) -> MilvusException:
    """构造一个模拟连接失败的异常。"""
    return MilvusException(message=message)


def build_store(**client_kwargs: object) -> tuple[MilvusStore, FakeMilvusClient]:
    """构造适配器并返回它使用的假客户端，便于断言。"""
    client = FakeMilvusClient(**client_kwargs)  # type: ignore[arg-type]
    store = MilvusStore(make_config(), client_factory=lambda: client)
    return store, client


class TestClientCreation:
    def test_creation_failure_is_reported_not_raised(self) -> None:
        # 回归：MilvusClient 在构造时就建立连接，服务不可用会在构造处抛异常。
        # 因此连接必须延后到 health_check 内部建立，否则适配器一构造就崩，
        # 异常处理根本没机会生效。
        def failing_factory() -> FakeMilvusClient:
            raise connect_error("Fail connecting to server")

        store = MilvusStore(make_config(), client_factory=failing_factory)

        result = store.health_check()

        assert result.ok is False
        assert "Fail connecting" in result.detail

    def test_client_is_created_lazily(self) -> None:
        created: list[FakeMilvusClient] = []

        def factory() -> FakeMilvusClient:
            client = FakeMilvusClient()
            created.append(client)
            return client

        store = MilvusStore(make_config(), client_factory=factory)
        assert created == []

        store.health_check()
        assert len(created) == 1

        # 第二次探测复用同一个客户端，不重复建连
        store.health_check()
        assert len(created) == 1


class TestHealthCheckSuccess:
    def test_reports_ok_with_server_version(self) -> None:
        store, _ = build_store(version="2.6.6")

        result = store.health_check()

        assert result.ok is True
        assert result.service == "milvus"
        assert "2.6.6" in result.detail

    def test_reports_existing_database(self) -> None:
        store, _ = build_store(databases=["default", "cbe_compliance"])

        result = store.health_check()

        assert result.ok is True
        assert "已存在" in result.detail

    def test_reports_missing_database(self) -> None:
        # 服务连得上但业务库还没建，是部署初期最常见的情况。
        # 这不算连通性失败，但必须说清楚，否则使用者看到一片绿色
        # 会以为可以直接用了。
        store, _ = build_store(databases=["default"])

        result = store.health_check()

        assert result.ok is True
        assert "未创建" in result.detail

    def test_elapsed_time_is_recorded(self) -> None:
        store, _ = build_store()

        assert store.health_check().elapsed_ms >= 0.0


class TestHealthCheckFailure:
    def test_connection_failure_is_reported_not_raised(self) -> None:
        store, _ = build_store(fail_with=connect_error("connection refused"))

        result = store.health_check()

        assert result.ok is False
        assert "connection refused" in result.detail

    def test_list_databases_failure_is_reported_not_raised(self) -> None:
        # 版本号拿到了但列举库失败（例如权限不足），同样不能放行
        store, _ = build_store(list_fail_with=RuntimeError("permission denied"))

        result = store.health_check()

        assert result.ok is False
        assert "permission denied" in result.detail

    def test_unexpected_exception_is_reported_not_raised(self) -> None:
        # 回归：失败来源不止驱动异常，任何异常都不得穿透 health_check
        store, _ = build_store(fail_with=ValueError("boom"))

        result = store.health_check()

        assert result.ok is False
        assert "ValueError" in result.detail


class TestLifecycle:
    def test_health_check_does_not_close_the_client(self) -> None:
        # MilvusClient 内部维护连接，应当长期复用
        store, client = build_store()

        store.health_check()

        assert client.closed is False

    def test_close_releases_the_client(self) -> None:
        store, client = build_store()
        store.health_check()

        store.close()

        assert client.closed is True

    def test_close_before_any_use_is_safe(self) -> None:
        # 从未连接过就关闭，不应报错
        store, _ = build_store()

        store.close()


class TestHealthResultRendering:
    def test_ok_line_starts_with_ok(self) -> None:
        store, _ = build_store()

        assert store.health_check().render().startswith("[OK")

    def test_fail_line_is_marked(self) -> None:
        store, _ = build_store(fail_with=connect_error("refused"))

        assert store.health_check().render().startswith("[FAIL]")
