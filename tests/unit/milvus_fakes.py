"""Milvus 适配器测试共用的假对象与构造辅助。

单独放一个模块而不是留在某个测试文件里：集合管理与检索两组测试都要用，
从一个测试文件 import 另一个会让「哪个是测试、哪个是辅助」变含糊。
"""

from __future__ import annotations

from typing import Any

from pymilvus.client.types import LoadState
from pymilvus.exceptions import MilvusException

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.records import ChunkVector


def make_config() -> MilvusConfig:
    return MilvusConfig(
        host="192.168.88.101",
        port=19530,
        database="cbe_compliance",
        collection="cbe_chunks_v1",
    )


class FakeMilvusClient:
    """假的 MilvusClient（不指定 db_name 的那个）。

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
    """假的、连接到业务库的客户端。

    每个方法都把调用参数记下来，测试按需断言。查询类方法的返回值由
    构造参数注入——返回什么由测试决定，假对象不猜。
    """

    def __init__(
        self,
        collections: tuple[str, ...] = (),
        query_rows: tuple[dict[str, object], ...] = (),
        write_fail_with: Exception | None = None,
        load_state: LoadState = LoadState.NotLoad,
        search_rows: tuple[dict[str, object], ...] = (),
        hybrid_rows: tuple[dict[str, object], ...] = (),
    ) -> None:
        self._collections = list(collections)
        self._query_rows = list(query_rows)
        self._write_fail_with = write_fail_with
        self._load_state = load_state
        self._search_rows = list(search_rows)
        self._hybrid_rows = list(hybrid_rows)
        self.created: list[tuple[str, FakeSchema]] = []
        self.indexed: list[tuple[str, FakeIndexParams]] = []
        self.schema_kwargs: dict[str, object] = {}
        self.upserted: list[list[dict[str, object]]] = []
        self.deleted: list[tuple[str, str]] = []
        self.queried: list[tuple[str, str, list[str]]] = []
        self.load_states_asked: list[str] = []
        self.loaded: list[str] = []
        self.searched: list[dict[str, object]] = []
        self.hybrid_searched: list[dict[str, object]] = []
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

    def get_load_state(
        self, collection_name: str, **kwargs: object
    ) -> dict[str, object]:
        self.load_states_asked.append(collection_name)
        return {"state": self._load_state}

    def load_collection(self, collection_name: str, **kwargs: object) -> None:
        self.loaded.append(collection_name)
        self._load_state = LoadState.Loaded

    def upsert(
        self, collection_name: str, data: list[dict[str, object]], **kwargs: object
    ) -> dict[str, int]:
        if self._write_fail_with is not None:
            raise self._write_fail_with
        self.upserted.append(list(data))
        return {"upsert_count": len(data)}

    def delete(
        self, collection_name: str, filter: str = "", **kwargs: object
    ) -> dict[str, int]:
        if self._write_fail_with is not None:
            raise self._write_fail_with
        self.deleted.append((collection_name, filter))
        return {"delete_count": 0}

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        self.queried.append((collection_name, filter, list(output_fields or [])))
        return [dict(row) for row in self._query_rows]

    def search(
        self,
        collection_name: str,
        data: list[object],
        anns_field: str,
        search_params: dict[str, object],
        limit: int,
        filter: str = "",
        **kwargs: object,
    ) -> list[list[dict[str, object]]]:
        self.searched.append(
            {
                "anns_field": anns_field,
                "search_params": search_params,
                "limit": limit,
                "filter": filter,
            }
        )
        # 外层多包一层：真实接口是「每个查询向量一个结果列表」，
        # 即使只传一个查询向量也如此
        return [self._search_rows]

    def hybrid_search(
        self,
        collection_name: str,
        reqs: list[object],
        ranker: object,
        limit: int,
        output_fields: list[str] | None = None,
        **kwargs: object,
    ) -> list[list[dict[str, object]]]:
        self.hybrid_searched.append(
            {
                "reqs": list(reqs),
                "ranker": ranker,
                "limit": limit,
                "output_fields": list(output_fields or []),
            }
        )
        return [self._hybrid_rows]

    def close(self) -> None:
        self.closed = True


def build_db_store(
    collections: tuple[str, ...] = (),
    query_rows: tuple[dict[str, object], ...] = (),
    write_fail_with: Exception | None = None,
    load_state: LoadState = LoadState.NotLoad,
    search_rows: tuple[dict[str, object], ...] = (),
    hybrid_rows: tuple[dict[str, object], ...] = (),
) -> tuple[MilvusStore, FakeMilvusDbClient, FakeMilvusClient]:
    """构造注入了「库客户端」与「普通客户端」的适配器。

    两个客户端分开注入，才能验证集合操作走的是哪一个。
    """
    db_client = FakeMilvusDbClient(
        collections,
        query_rows,
        write_fail_with,
        load_state,
        search_rows,
        hybrid_rows,
    )
    plain_client = FakeMilvusClient()
    store = MilvusStore(
        make_config(),
        client_factory=lambda: plain_client,
        db_client_factory=lambda: db_client,
    )
    return store, db_client, plain_client


def make_vector(**overrides: Any) -> ChunkVector:
    """造一条待写入的子块向量。"""
    fields: dict[str, Any] = {
        "chunk_id": "doc-1_c0000",
        "doc_id": "doc-1",
        "parent_id": "doc-1_p0000",
        "chunk_index": 0,
        "country": "EU",
        "doc_type": "policy",
        "publisher": "amazon",
        "dense": [0.1, 0.2],
        "sparse": {7: 0.5, 42: 0.25},
    }
    fields.update(overrides)
    return ChunkVector(**fields)


def make_query_row(**overrides: Any) -> dict[str, Any]:
    """造一行 Milvus 查询结果，含两路向量。"""
    row: dict[str, Any] = {
        "chunk_id": "doc-1_c0000",
        "doc_id": "doc-1",
        "parent_id": "doc-1_p0000",
        "chunk_index": 0,
        "dense_vector": [0.1, 0.2],
        "sparse_vector": {7: 0.5},
    }
    row.update(overrides)
    return row


def make_hit(
    chunk_id: str = "doc-1_c0000",
    doc_id: str = "doc-1",
    parent_id: str = "doc-1_p0000",
    distance: float = 0.6,
) -> dict[str, Any]:
    """造一条检索命中，形状与 pymilvus 的返回一致。"""
    return {
        "chunk_id": chunk_id,
        "distance": distance,
        "entity": {
            "doc_id": doc_id,
            "parent_id": parent_id,
            "chunk_id": chunk_id,
        },
    }


def request_details(requests: list[object]) -> list[dict[str, object]]:
    """把 AnnSearchRequest 摊成普通字典，便于断言。

    pymilvus 把要用的东西都放在下划线属性里，没有公开的读取方式。
    摊在这里，测试就不必到处 getattr。
    """
    return [
        {
            "anns_field": getattr(request, "_anns_field", None),
            "param": getattr(request, "_param", None),
            "limit": getattr(request, "_limit", None),
            "expr": getattr(request, "_expr", None),
        }
        for request in requests
    ]
