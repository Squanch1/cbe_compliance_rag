"""Milvus 适配器。

业务层通过本类访问 Milvus，不直接 import pymilvus 驱动
（见 docs/spec/02-architecture.md 第 2.4 节）。

与另外三个适配器的关键差异：MilvusClient 在构造时就建立连接，
服务不可用会在构造处抛异常。因此本适配器的客户端延迟到首次使用时
才创建，让连接失败落在 health_check 的异常处理范围内。
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Protocol

from pymilvus import AnnSearchRequest, DataType, MilvusClient, WeightedRanker
from pymilvus.client.types import LoadState

from cbe_rag.config.settings import MilvusConfig
from cbe_rag.storage.ddl import (
    MILVUS_DENSE_FIELD,
    MILVUS_DENSE_INDEX,
    MILVUS_FIELDS,
    MILVUS_SCALAR_INDEX_FIELDS,
    MILVUS_SCALAR_INDEX_TYPE,
    MILVUS_SPARSE_FIELD,
    MILVUS_SPARSE_INDEX,
)
from cbe_rag.storage.health import HealthResult
from cbe_rag.storage.records import ChunkVector, VectorHit


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

    def load_collection(self, collection_name: str, **kwargs: Any) -> Any:
        ...

    def get_load_state(self, collection_name: str, **kwargs: Any) -> dict[str, Any]:
        ...

    def upsert(
        self, collection_name: str, data: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        ...

    def delete(
        self, collection_name: str, filter: str = "", **kwargs: Any
    ) -> Any:
        ...

    def query(
        self,
        collection_name: str,
        filter: str = "",
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        ...

    def search(
        self,
        collection_name: str,
        data: list[Any],
        anns_field: str,
        search_params: dict[str, Any],
        limit: int,
        filter: str = "",
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        ...

    def hybrid_search(
        self,
        collection_name: str,
        reqs: list[AnnSearchRequest],
        ranker: WeightedRanker,
        limit: int,
        output_fields: list[str] | None = None,
        **kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        ...

    def close(self) -> None:
        ...


# doc_id 与 chunk_id 的形状：UUID，或 UUID 加 _p0000 / _c0000 后缀。
# Milvus 的 filter 表达式没有参数占位符，值只能拼进字符串，因此拼之前
# 先校验形状。两个值都是程序生成的，正常不会含引号；校验是防止调用方
# 传进别的东西，把过滤条件整个改了。
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _doc_id_filter(doc_id: str) -> str:
    """拼出按 doc_id 过滤的表达式。"""
    if not _SAFE_ID.match(doc_id):
        raise ValueError("doc_id 含非法字符，无法安全拼进 filter：%r" % doc_id)
    return 'doc_id == "%s"' % doc_id


# 回写标量字段时必须一并带上的列。
#
# Milvus 的 upsert 是**整行替换**，没给的列会被清空——尤其不能漏掉两路
# 向量，否则元数据改一次向量就没了，文档从此检索不到，而且不报错。
_SCALAR_UPDATE_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "doc_id",
    "parent_id",
    "chunk_index",
    "dense_vector",
    "sparse_vector",
)


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
        # 集合是否已确认加载到内存，见 _ensure_loaded
        self._loaded = False

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

    def _ensure_loaded(self) -> None:
        """确保集合已加载到内存。

        **建好集合与索引还不够。** delete 与 query 在未加载的集合上会直接
        报 `collection not loaded`，而 upsert 不需要加载——所以漏掉这一步
        时的症状很怪：向量写进去了，删的时候才炸，还在库里留下脏数据。

        带实例级缓存，只查一次状态。
        """
        if self._loaded:
            return
        client = self._get_db_client()
        state = client.get_load_state(self._config.collection)
        if state.get("state") != LoadState.Loaded:
            client.load_collection(self._config.collection)
        self._loaded = True

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
        schema.add_field(MILVUS_DENSE_FIELD, DataType.FLOAT_VECTOR, dim=dense_dim)
        for name, type_name, params in MILVUS_FIELDS:
            schema.add_field(name, getattr(DataType, type_name), **params)
        client.create_collection(self._config.collection, schema=schema)

        # 稠密用精确检索，稀疏用倒排，三个过滤维度各建倒排索引。
        # 索引参数直接取自 ddl，避免与文档里的定义两处写死。
        index_params = client.prepare_index_params()
        dense_type, dense_metric = MILVUS_DENSE_INDEX
        index_params.add_index(
            field_name=MILVUS_DENSE_FIELD,
            index_type=dense_type,
            metric_type=dense_metric,
        )
        sparse_type, sparse_metric = MILVUS_SPARSE_INDEX
        index_params.add_index(
            field_name=MILVUS_SPARSE_FIELD,
            index_type=sparse_type,
            metric_type=sparse_metric,
        )
        for field_name in MILVUS_SCALAR_INDEX_FIELDS:
            index_params.add_index(
                field_name=field_name, index_type=MILVUS_SCALAR_INDEX_TYPE
            )
        client.create_index(self._config.collection, index_params=index_params)
        # 建完还要加载到内存，否则 delete 与 query 用不了，理由见 _ensure_loaded
        client.load_collection(self._config.collection)
        self._loaded = True

        return True

    def upsert_chunks(self, vectors: list[ChunkVector]) -> int:
        """写入子块向量，返回写入条数。空列表直接返回，不发请求。

        用 upsert 而不是 insert：重跑同一份文档时 chunk_id 不变，
        insert 会撞主键。

        **但重跑不能只靠 upsert。** 新的分块可能比旧的少，多出来的
        那几条会留在库里继续被检索到——召回的是已经被删掉的段落。
        调用方要先调 delete_by_doc 清干净。
        """
        if not vectors:
            return 0

        self._ensure_loaded()
        rows = [
            {
                "chunk_id": vector.chunk_id,
                "dense_vector": vector.dense,
                "sparse_vector": vector.sparse,
                "doc_id": vector.doc_id,
                "parent_id": vector.parent_id,
                "chunk_index": vector.chunk_index,
                "country": vector.country,
                "doc_type": vector.doc_type,
                "publisher": vector.publisher,
            }
            for vector in vectors
        ]
        self._get_db_client().upsert(
            collection_name=self._config.collection, data=rows
        )
        return len(rows)

    def delete_by_doc(self, doc_id: str) -> None:
        """删掉某份文档的全部向量。

        重跑这份文档前调用，理由见 upsert_chunks。
        """
        self._ensure_loaded()
        self._get_db_client().delete(
            collection_name=self._config.collection,
            filter=_doc_id_filter(doc_id),
        )

    def update_scalar_fields(
        self, doc_id: str, *, country: str, doc_type: str, publisher: str
    ) -> int:
        """把某文档所有子块的过滤字段改成新值，返回改了几条。

        做法是查出来、改字段、再整行写回。**必须连两路向量一起写回**：
        Milvus 的 upsert 是整行替换，只给标量字段会把向量抹掉，文档
        从此检索不到，而且不报错。

        代价是一次读加一次写。只发生在「内容没变但清单上的元数据改了」
        这一种情况下，且同一文档的子块数量有限，可以接受。
        """
        self._ensure_loaded()
        client = self._get_db_client()
        rows = client.query(
            collection_name=self._config.collection,
            filter=_doc_id_filter(doc_id),
            output_fields=list(_SCALAR_UPDATE_FIELDS),
        )
        if not rows:
            return 0

        updated = [
            {
                "chunk_id": row["chunk_id"],
                "doc_id": row["doc_id"],
                "parent_id": row["parent_id"],
                "chunk_index": row["chunk_index"],
                "dense_vector": row["dense_vector"],
                "sparse_vector": row["sparse_vector"],
                "country": country,
                "doc_type": doc_type,
                "publisher": publisher,
            }
            for row in rows
        ]
        client.upsert(collection_name=self._config.collection, data=updated)
        return len(updated)

    def hybrid_search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        *,
        dense_limit: int,
        sparse_limit: int,
        weights: tuple[float, float],
        limit: int,
        filter_expression: str = "",
    ) -> list[VectorHit]:
        """两路召回后融合，返回融合分从高到低的命中。

        **两路的候选数通常不同**（默认 20 与 40），这是实测定的：稀疏路的
        分数分布很陡，头部之外基本是噪声，但头部偶尔能捞出稠密路完全找
        不到的专有名词，多取一些给融合一次机会。

        weights 是 (稠密权重, 稀疏权重)。融合前两路各自归一化——它们的
        量纲本来就不可比（同一个问题下稠密 0.5867、稀疏 0.0786）。

        结果为空表示没有任何命中，不是错误。
        """
        if limit <= 0:
            return []

        self._ensure_loaded()
        requests = [
            AnnSearchRequest(
                data=[dense],
                anns_field=MILVUS_DENSE_FIELD,
                param={"metric_type": MILVUS_DENSE_INDEX[1]},
                limit=dense_limit,
                filter=filter_expression or None,
            ),
            AnnSearchRequest(
                data=[sparse],
                anns_field=MILVUS_SPARSE_FIELD,
                param={"metric_type": MILVUS_SPARSE_INDEX[1]},
                limit=sparse_limit,
                filter=filter_expression or None,
            ),
        ]

        results = self._get_db_client().hybrid_search(
            collection_name=self._config.collection,
            reqs=requests,
            ranker=WeightedRanker(*weights),
            limit=limit,
            # parent_id 必须一并取回：折叠要靠它，而子块的编号是全局
            # 递增的，从 chunk_id 推不出它属于哪个父块
            output_fields=["chunk_id", "doc_id", "parent_id"],
        )
        if not results or not results[0]:
            return []

        return [
            VectorHit(
                chunk_id=hit["chunk_id"],
                doc_id=hit["entity"]["doc_id"],
                parent_id=hit["entity"]["parent_id"],
                score=float(hit["distance"]),
            )
            for hit in results[0]
        ]

    def top_dense_score(
        self, dense: list[float], *, filter_expression: str = ""
    ) -> float | None:
        """取稠密路的最高余弦相似度，没有命中时返回 None。

        **拒答判据用的是它，不是融合分。** 余弦相似度有绝对含义（0.6
        就是 0.6），而融合分经过归一化与加权，数值随权重变化——同一份
        语料在 0.7/0.3 下给 0.6970、0.5/0.5 下给 0.6407，拿它当阈值
        没有可比性。

        单独跑一次 limit=1 的检索：hybrid_search 的返回里只有融合分，
        没有各路的原始分。
        """
        self._ensure_loaded()
        results = self._get_db_client().search(
            collection_name=self._config.collection,
            data=[dense],
            anns_field=MILVUS_DENSE_FIELD,
            search_params={"metric_type": MILVUS_DENSE_INDEX[1]},
            limit=1,
            filter=filter_expression,
        )
        if not results or not results[0]:
            return None
        return float(results[0][0]["distance"])

    def close(self) -> None:
        """释放客户端。从未连接过时调用是安全的。"""
        for client in (self._client, self._db_client):
            if client is not None:
                client.close()
        self._client = None
        self._db_client = None
