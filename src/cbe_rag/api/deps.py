"""依赖注入。

适配器在应用启动时建一次、关闭时释放，请求处理中复用。每次请求新建是
行不通的：Milvus 客户端要握手，嵌入模型要常驻显存。
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from cbe_rag.config.settings import Settings
from cbe_rag.services.qa import QaContext
from cbe_rag.storage import (
    BailianClient,
    EmbeddingStore,
    MilvusStore,
    MysqlStore,
    RedisStore,
)


class ApiError(Exception):
    """接口层的错误，带契约里约定的错误码。

    路由抛这个，由 app.py 注册的处理器转成统一的错误响应形状。不直接用
    HTTPException 是因为它的 detail 结构随意，而契约要求所有非 2xx 响应
    共用一个形状。
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass
class AppState:
    """应用级的状态：配置与各适配器。

    由 lifespan 建好放进 app.state，请求处理时取出。
    """

    settings: Settings
    mysql: MysqlStore
    milvus: MilvusStore
    embedding: EmbeddingStore
    client: BailianClient
    redis: RedisStore

    def qa_context(self) -> QaContext:
        """组装问答用例需要的依赖。"""
        return QaContext(
            mysql=self.mysql,
            milvus=self.milvus,
            embedding=self.embedding,
            client=self.client,
            redis=self.redis,
            config=self.settings.retrieval,
        )

    def close(self) -> None:
        """释放各适配器。

        逐个关、单独吞异常：一个关不掉不该影响其余的，而 close 时抛出的
        异常也没人会处理——它发生在进程退出路径上。
        """
        for adapter in (
            self.mysql,
            self.milvus,
            self.embedding,
            self.client,
            self.redis,
        ):
            try:
                adapter.close()
            except Exception:
                pass


def build_state(settings: Settings) -> AppState:
    """按配置建好各适配器。构造它们不产生网络调用。"""
    return AppState(
        settings=settings,
        mysql=MysqlStore(settings.mysql),
        milvus=MilvusStore(settings.milvus),
        embedding=EmbeddingStore(settings.embedding),
        client=BailianClient(settings.llm),
        redis=RedisStore(settings.redis),
    )


def get_state(request: Request) -> AppState:
    """从应用状态里取 AppState。"""
    return request.app.state.app_state
