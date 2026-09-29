"""Milvus 适配器的单元测试。

注入假的客户端工厂，不连接真实服务。
"""

from __future__ import annotations

from pymilvus.exceptions import MilvusException

from pymilvus import DataType

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.ddl import (
    MILVUS_DENSE_INDEX,
    MILVUS_FIELDS,
    MILVUS_SCALAR_INDEX_FIELDS,
    MILVUS_SCALAR_INDEX_TYPE,
    MILVUS_SPARSE_INDEX,
)
from cbe_rag.storage.milvus_store import MilvusStore


def make_config() -> MilvusConfig:
    return MilvusConfig(
        host="192.168.88.101",
        port=19530,
        database="cbe_compliance",
        collection="cbe_chunks_v1",
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


class FakeSchema:
    """假的 schema 构造器，记录加进来的字段。"""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.fields: list[tuple[str, object, dict[str, object]]] = []

    def add_field(self, name: str, data_type: object, **params: object) -> None:
        self.fields.append((name, data_type, params))


class FakeIndexParams:
    """假的索引参数构造器，记录加进来的索引。"""

    def __init__(self) -> None:
        self.indexes: list[tuple[str, str | None, str | None]] = []

    def add_index(
        self,
        field_name: str,
        index_type: str | None = None,
        metric_type: str | None = None,
    ) -> None:
        self.indexes.append((field_name, index_type, metric_type))


class FakeMilvusDbClient:
    """假的、连接到业务库的客户端。"""

    def __init__(self, collections: tuple[str, ...] = ()) -> None:
        self._collections = list(collections)
        self.created: list[tuple[str, FakeSchema]] = []
        self.indexed: list[tuple[str, FakeIndexParams]] = []
        self.schema_kwargs: dict[str, object] = {}
        self.closed = False

    def list_collections(self) -> list[str]:
        return list(self._collections)

    def create_schema(self, **kwargs: object) -> FakeSchema:
        self.schema_kwargs = kwargs
        return FakeSchema(**kwargs)

    def prepare_index_params(self) -> FakeIndexParams:
        return FakeIndexParams()

    def create_collection(self, collection_name: str, **kwargs: object) -> None:
        self.created.append((collection_name, kwargs["schema"]))  # type: ignore[arg-type]

    def create_index(self, collection_name: str, **kwargs: object) -> None:
        self.indexed.append((collection_name, kwargs["index_params"]))  # type: ignore[arg-type]

    def close(self) -> None:
        self.closed = True


def build_db_store(
    collections: tuple[str, ...] = (),
) -> tuple[MilvusStore, FakeMilvusDbClient, FakeMilvusClient]:
    """构造注入了「库客户端」与「普通客户端」的适配器。

    两个客户端分开注入，才能验证集合操作走的是哪一个。
    """
    db_client = FakeMilvusDbClient(collections)
    plain_client = FakeMilvusClient()
    store = MilvusStore(
        make_config(),
        client_factory=lambda: plain_client,
        db_client_factory=lambda: db_client,
    )
    return store, db_client, plain_client


class TestCreateCollection:
    def test_creates_when_absent(self) -> None:
        store, db_client, _ = build_db_store(collections=())

        assert store.create_collection(dense_dim=1024) is True
        assert [name for name, _ in db_client.created] == ["cbe_chunks_v1"]

    def test_skips_when_present(self) -> None:
        store, db_client, _ = build_db_store(collections=("cbe_chunks_v1",))

        assert store.create_collection(dense_dim=1024) is False
        assert db_client.created == []

    def test_uses_the_database_client(self) -> None:
        # 回归：方法级 db_name 参数会被 Milvus 静默忽略——实测
        # create_collection(name, schema, db_name="cbe_compliance")
        # 会把集合建到 default 库。必须走连接到业务库的客户端。
        store, db_client, plain_client = build_db_store()

        store.create_collection(dense_dim=1024)

        assert db_client.created, "应当使用库客户端"
        assert not plain_client.closed, "普通客户端不该被用到"

    def test_disables_auto_id_and_dynamic_fields(self) -> None:
        # auto_id 会覆盖我们自定的 chunk_id；动态字段会让 schema 失控
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        assert db_client.schema_kwargs["auto_id"] is False
        assert db_client.schema_kwargs["enable_dynamic_field"] is False


class TestCollectionSchema:
    def test_all_defined_fields_are_added(self) -> None:
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        schema = db_client.created[0][1]
        added = [name for name, _, _ in schema.fields]
        # 稠密向量由调用方传入维度，单独加，因此顺序上排在最前
        assert added[0] == "dense_vector"
        assert set(added) == {name for name, _, _ in MILVUS_FIELDS} | {"dense_vector"}

    def test_chunk_id_is_the_primary_key(self) -> None:
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        schema = db_client.created[0][1]
        primary = [f for f in schema.fields if f[2].get("is_primary")]
        assert len(primary) == 1
        assert primary[0][0] == "chunk_id"
        assert primary[0][1] is DataType.VARCHAR

    def test_dense_dimension_comes_from_the_argument(self) -> None:
        # 维度必须与嵌入模型一致，因此由调用方传入而不是写死在字段表里。
        # 写死会形成第二个真相来源，改了嵌入配置而忘了改这里就静默不匹配。
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=768)

        schema = db_client.created[0][1]
        dense = next(f for f in schema.fields if f[0] == "dense_vector")
        assert dense[1] is DataType.FLOAT_VECTOR
        assert dense[2]["dim"] == 768

    def test_dense_dimension_is_never_in_the_field_table(self) -> None:
        # 字段表里不该出现 dense_vector，否则会与传入的维度冲突
        assert "dense_vector" not in {name for name, _, _ in MILVUS_FIELDS}

    def test_sparse_vector_has_no_dimension(self) -> None:
        # 稀疏向量的维度由词元索引隐式决定，指定了反而报错
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        schema = db_client.created[0][1]
        sparse = next(f for f in schema.fields if f[0] == "sparse_vector")
        assert sparse[1] is DataType.SPARSE_FLOAT_VECTOR
        assert "dim" not in sparse[2]


class TestCollectionIndexes:
    def test_dense_uses_configured_index_and_metric(self) -> None:
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        indexes = db_client.indexed[0][1].indexes
        dense = next(i for i in indexes if i[0] == "dense_vector")
        assert (dense[1], dense[2]) == MILVUS_DENSE_INDEX

    def test_sparse_uses_inner_product(self) -> None:
        # 稀疏向量只支持内积
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        indexes = db_client.indexed[0][1].indexes
        sparse = next(i for i in indexes if i[0] == "sparse_vector")
        assert (sparse[1], sparse[2]) == MILVUS_SPARSE_INDEX

    def test_every_filter_field_gets_a_scalar_index(self) -> None:
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        indexes = db_client.indexed[0][1].indexes
        indexed_fields = {i[0] for i in indexes}
        for field_name in MILVUS_SCALAR_INDEX_FIELDS:
            assert field_name in indexed_fields

    def test_scalar_indexes_use_inverted_type(self) -> None:
        store, db_client, _ = build_db_store()

        store.create_collection(dense_dim=1024)

        indexes = db_client.indexed[0][1].indexes
        for field_name, index_type, _ in indexes:
            if field_name in MILVUS_SCALAR_INDEX_FIELDS:
                assert index_type == MILVUS_SCALAR_INDEX_TYPE


class TestCloseReleasesBothClients:
    def test_close_releases_database_client(self) -> None:
        store, db_client, _ = build_db_store()
        store.create_collection(dense_dim=1024)

        store.close()

        assert db_client.closed is True

    def test_close_releases_both_when_both_were_used(self) -> None:
        # 两个客户端都是懒创建的：没用到就不会建，close 也就不该去关它。
        # 这里先让两个都建起来，再验证 close 把它们都释放。
        store, db_client, plain_client = build_db_store()
        store.health_check()
        store.create_collection(dense_dim=1024)

        store.close()

        assert db_client.closed is True
        assert plain_client.closed is True

    def test_close_does_not_touch_unused_client(self) -> None:
        # 从未使用的客户端不该被关闭——适配器只释放自己创建的东西
        store, _, plain_client = build_db_store()
        store.create_collection(dense_dim=1024)

        store.close()

        assert plain_client.closed is False
