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

from pymilvus import DataType, MilvusClient

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.ddl import (
    MILVUS_DENSE_INDEX,
    MILVUS_FIELDS,
    MILVUS_SCALAR_INDEX_FIELDS,
    MILVUS_SCALAR_INDEX_TYPE,
    MILVUS_SPARSE_INDEX,
)
from cbe_rag.storage.health import HealthResult


class MilvusClientProtocol(Protocol):
    """MilvusClient 中本适配器用到的部分。

    单独声明是为了让单元测试能注入假实现，不必连接真实服务。
    """

    def get_server_version(self) -> str:
        ...

    def list_databases(self) -> list[str]:
        ...

    def list_collections(self) -> list[str]:
        ...

    def create_database(self, database_name: str) -> Any:
        ...

    def create_schema(self, **kwargs: Any) -> Any:
        ...

    def prepare_index_params(self) -> Any:
        ...

    def create_collection(self, collection_name: str, **kwargs: Any) -> Any:
        ...

    def create_index(self, collection_name: str, **kwargs: Any) -> Any:
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
        db_client_factory: Callable[[], MilvusClientProtocol] | None = None,
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
        self._db_client_factory: Callable[[], MilvusClientProtocol] = (
            db_client_factory
            if db_client_factory is not None
            else self._connect_db_by_config
        )
        self._client: MilvusClientProtocol | None = None
        self._db_client: MilvusClientProtocol | None = None

    def _connect_by_config(self) -> MilvusClientProtocol:
        """按配置建立真实客户端。

        不指定 db_name，连到服务端默认库。业务库尚未创建时也能完成连通性检查，
        而指定 db_name 会让这种情况直接失败。
        """
        return MilvusClient(
            uri="http://%s:%d" % (self._config.host, self._config.port)
        )

    def _connect_db_by_config(self) -> MilvusClientProtocol:
        """建立连接到**业务库**的客户端。

        **必须在这里指定 db_name，不能在调用方法时传。** MilvusClient
        的方法级 db_name 参数会被静默忽略——实测
        create_collection(name, schema, db_name="cbe_compliance")
        会把集合建到 default 库且不报错，而 default 库里有其他项目的
        demo_v8，正是 CLAUDE.md 明令禁止写入的地方。
        """
        return MilvusClient(
            uri="http://%s:%d" % (self._config.host, self._config.port),
            db_name=self._config.database,
        )

    def _get_client(self) -> MilvusClientProtocol:
        """返回已建立的客户端，没有就先建一个。"""
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _get_db_client(self) -> MilvusClientProtocol:
        """返回连接到业务库的客户端，没有就先建一个。"""
        if self._db_client is None:
            self._db_client = self._db_client_factory()
        return self._db_client

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

    def ensure_database(self) -> bool:
        """确保业务库存在，返回是否真的创建了。

        用**不指定 db_name** 的客户端：目标库还不存在时，
        带 db_name 的客户端连上去会直接失败。
        """
        client = self._get_client()
        if self._config.database in client.list_databases():
            return False
        client.create_database(self._config.database)
        return True

    def create_collection(self, dense_dim: int) -> bool:
        """创建向量集合，已存在则跳过。

        dense_dim 是稠密向量的维度，**必须与嵌入模型一致**，因此由
        调用方按嵌入配置传入，不在这里写死。

        返回是否真的创建了（已存在时为 False），便于脚本区分
        「这次真的建了」和「本来就有」。

        **必须用连接到业务库的客户端**，理由见 _connect_db_by_config。
        """
        client = self._get_db_client()

        if self._config.collection in client.list_collections():
            return False

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        # 稠密向量单独加：维度来自嵌入配置而不是字段表
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dense_dim)
        for name, type_name, params in MILVUS_FIELDS:
            schema.add_field(name, getattr(DataType, type_name), **params)
        client.create_collection(self._config.collection, schema=schema)

        # 稠密用精确检索，稀疏用倒排，三个过滤维度各建倒排索引。
        # 索引参数直接取自 ddl，避免与文档里的定义两处写死。
        index_params = client.prepare_index_params()
        dense_type, dense_metric = MILVUS_DENSE_INDEX
        index_params.add_index(
            field_name="dense_vector", index_type=dense_type, metric_type=dense_metric
        )
        sparse_type, sparse_metric = MILVUS_SPARSE_INDEX
        index_params.add_index(
            field_name="sparse_vector", index_type=sparse_type, metric_type=sparse_metric
        )
        for field_name in MILVUS_SCALAR_INDEX_FIELDS:
            index_params.add_index(
                field_name=field_name, index_type=MILVUS_SCALAR_INDEX_TYPE
            )
        client.create_index(self._config.collection, index_params=index_params)

        return True

    def close(self) -> None:
        """释放客户端。从未连接过时调用是安全的。"""
        for client in (self._client, self._db_client):
            if client is not None:
                client.close()
        self._client = None
        self._db_client = None
