"""Milvus 适配器。

业务层通过本类访问 Milvus，不直接 import pymilvus 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。

与另外三个适配器的关键差异：MilvusClient 在构造时就建立连接，
服务不可用会在构造处抛异常。因此本适配器的客户端延迟到首次使用时
才创建，让连接失败落在 health_check 的异常处理范围内。
"""

from __future__ import annotations

import time
from typing import Callable, Protocol

from pymilvus import MilvusClient

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.health import HealthResult


class MilvusClientProtocol(Protocol):
    """MilvusClient 中本适配器用到的部分。

    单独声明是为了让单元测试能注入假实现，不必连接真实服务。
    """

    def get_server_version(self) -> str:
        ...

    def list_databases(self) -> list[str]:
        ...

    def close(self) -> None:
        ...


class MilvusStore:
    """Milvus 访问入口。

    构造本对象不产生网络调用；连接在第一次实际操作时建立并复用。
    """

    def __init__(
        self,
        config: MilvusConfig,
        client_factory: Callable[[], MilvusClientProtocol] | None = None,
    ) -> None:
        """初始化。

        client_factory 仅用于测试注入；生产路径下由配置构造真实客户端。
        之所以注入工厂而不是现成客户端，是因为客户端构造本身就可能失败，
        必须把这一步也推迟到 health_check 内部。
        """
        self._config = config
        self._client_factory: Callable[[], MilvusClientProtocol] = (
            client_factory if client_factory is not None else self._connect_by_config
        )
        self._client: MilvusClientProtocol | None = None

    def _connect_by_config(self) -> MilvusClientProtocol:
        """按配置建立真实客户端。

        不指定 db_name，连到服务端默认库。业务库尚未创建时也能完成连通性检查，
        而指定 db_name 会让这种情况直接失败。
        """
        return MilvusClient(
            uri="http://%s:%d" % (self._config.host, self._config.port)
        )

    def _get_client(self) -> MilvusClientProtocol:
        """返回已建立的客户端，没有就先建一个。"""
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def health_check(self) -> HealthResult:
        """探测 Milvus 连通性，并报告业务库是否已创建。

        业务库未创建不算连通性失败——服务本身是好的，只是还没初始化。
        但必须在结果里写明，否则使用者看到通过会以为可以直接使用。
        """
        started = time.perf_counter()
        try:
            client = self._get_client()
            version = client.get_server_version()
            databases = client.list_databases()
            exists = self._config.database in databases
            ok = True
            detail = "version=%s database=%s(%s)" % (
                version,
                self._config.database,
                "已存在" if exists else "未创建，需先初始化",
            )
        except Exception as exc:
            # 刻意捕获所有异常。健康检查的契约是报告问题而非抛异常，
            # 失败来源不止驱动异常，漏掉一种就变成调用方崩溃。
            ok = False
            detail = "%s: %s" % (type(exc).__name__, exc)

        return HealthResult(
            service="milvus",
            ok=ok,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放客户端。从未连接过时调用是安全的。"""
        if self._client is not None:
            self._client.close()
            self._client = None
