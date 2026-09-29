"""HTTP 路由。

只做三件事：校验请求、调用例层、转响应。业务规则一律不写在这里
（见 02-architecture 第 7 节）。
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from cbe_rag.api.deps import ApiError, AppState, get_state
from cbe_rag.api.schemas import (
    AskRequest,
    AskResponse,
    DimensionOut,
    DimensionsOut,
    HealthItemOut,
    HealthResponse,
    ParentOut,
    to_ask_response,
)
from cbe_rag.storage.records import Dimension
from cbe_rag.services.qa import QaRequest, answer
from cbe_rag.storage import (
    BailianClient,
    EmbeddingStore,
    MilvusStore,
    MysqlStore,
    RedisStore,
)

router = APIRouter(prefix="/api/v1")

State = Annotated[AppState, Depends(get_state)]


def _filters_of(payload: AskRequest) -> dict[str, str | None]:
    """把请求里的过滤条件摊平，交给用例层。"""
    filters = payload.filters
    if filters is None:
        return {"country": None, "doc_type": None, "publisher": None}
    return {
        "country": filters.country,
        "doc_type": filters.doc_type,
        "publisher": filters.publisher,
    }


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request, state: State) -> AskResponse:
    """提问。

    trace_id 由中间件生成并放进 request.state，这里取出来交给用例层——
    它进日志、也进响应，使用者报问题时能直接给出这个值。
    """
    trace_id = getattr(request.state, "trace_id", None) or str(uuid.uuid4())
    result = answer(
        QaRequest(question=payload.question, session_id=payload.session_id, **_filters_of(payload)),
        state.qa_context(),
        trace_id=trace_id,
    )
    return to_ask_response(result)


def _to_dimension(item: Dimension) -> DimensionOut:
    """把存储层的维度记录摊成响应模型。"""
    return DimensionOut(
        code=item.code, name_zh=item.name_zh, name_en=item.name_en
    )


@router.get("/dimensions", response_model=DimensionsOut)
def get_dimensions(state: State) -> DimensionsOut:
    """取三张维度表的可用取值。

    界面靠它把 `EU`、`guideline` 这类代码显示成「欧盟」「官方指南」。
    代码是后端筛选用具，直接摆给使用者看等于让人先学一遍术语表。

    **取值来自维度表，不是写死的。** 以后扩充国家或文档类型只改数据，
    界面自动跟着变。
    """
    dimensions = state.mysql.list_dimensions()
    return DimensionsOut(
        countries=[_to_dimension(item) for item in dimensions.countries],
        doc_types=[_to_dimension(item) for item in dimensions.doc_types],
        publishers=[_to_dimension(item) for item in dimensions.publishers],
    )


@router.get("/parents/{parent_id}", response_model=ParentOut)
def get_parent(parent_id: str, state: State) -> ParentOut:
    """按 parent_id 取父块全文，供界面「展开上下文」时调用。

    父块不存在（比如界面拿着过期的 id 来查）返回 404，这是使用者的正常
    操作路径上的情况，不是服务出错。
    """
    chunks = state.mysql.get_chunks([parent_id])
    if not chunks:
        raise ApiError(404, "unknown_parent", "找不到这个父块：%s" % parent_id)

    chunk = chunks[0]
    documents = state.mysql.get_documents([chunk.doc_id])
    if not documents:
        raise ApiError(
            404, "unknown_parent", "父块所属的文档记录也找不到：%s" % chunk.doc_id
        )

    record = documents[0]
    return ParentOut(
        parent_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        token_count=chunk.token_count,
        chunk_index=chunk.chunk_index,
        title=record.title,
        source_url=record.source_url,
        effective_date=record.effective_date,
        collected_date=record.collected_date,
        country=record.country,
        doc_type=record.doc_type,
        publisher=record.publisher,
    )


def _health_items(state: AppState) -> list[HealthItemOut]:
    """逐个探测各依赖。

    每个适配器的 health_check 都吞掉自己的异常、返回结果对象，因此这里
    不需要 try——这是它们的契约（见 storage/health.py）。
    """
    adapters: list[
        MysqlStore | MilvusStore | RedisStore | EmbeddingStore | BailianClient
    ] = [
        state.milvus,
        state.mysql,
        state.redis,
        state.embedding,
        state.client,
    ]
    return [
        HealthItemOut(
            service=result.service,
            ok=result.ok,
            detail=result.detail,
            elapsed_ms=result.elapsed_ms,
        )
        for result in (adapter.health_check() for adapter in adapters)
    ]


@router.get("/health", response_model=HealthResponse)
def health(state: State) -> HealthResponse:
    """报告各依赖的连通性。

    任一依赖不通时 ok 为 False，但**状态码仍是 200**：这个接口的语义是
    「报告状态」，不是「服务本身能不能响应」。返回 503 会让监控把「某个
    依赖挂了」和「这个服务挂了」混为一谈。
    """
    services = _health_items(state)
    return HealthResponse(
        ok=all(item.ok for item in services), services=services
    )
