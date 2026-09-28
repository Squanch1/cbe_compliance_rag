"""外部服务适配层。

业务层只依赖本层暴露的接口，不直接 import 任何驱动库
（见 CLAUDE.md 5.3 与 docs/spec/02-architecture.md 第 2.4 节）。
"""

from cbe_rag.storage.embedding import EmbeddingResult, EmbeddingStore
from cbe_rag.storage.health import HealthResult, StorageAdapter
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.mongo_store import MongoStore
from cbe_rag.storage.mysql_store import MysqlStore
from cbe_rag.storage.redis_store import RedisStore

__all__ = [
    "EmbeddingResult",
    "EmbeddingStore",
    "HealthResult",
    "MilvusStore",
    "MongoStore",
    "MysqlStore",
    "RedisStore",
    "StorageAdapter",
]
