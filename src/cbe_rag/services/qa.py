"""问答用例。

一次完整的问答：取会话、查缓存、检索、生成、写回会话与缓存。

接口层调这一个函数就够，业务规则不散在路由里（见 02-architecture 第 7 节）。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.generation.answer import Answer
from cbe_rag.generation.citations import CitationReport
from cbe_rag.generation.service import answer_question
from cbe_rag.retrieval.models import (
    RetrievalQuery,
    RetrievedChild,
    RetrievedParent,
)
from cbe_rag.storage.bailian_client import BailianClient
from cbe_rag.storage.embedding import EmbeddingStore
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.mysql_store import MysqlStore
from cbe_rag.storage.redis_store import RedisStore

# 会话里消息的角色名，与提示词里的 role 取值一致。
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass(frozen=True)
class QaRequest:
    """一次问答请求。三个过滤维度由调用方显式传，不从问题里猜。"""

    question: str
    session_id: str | None = None
    country: str | None = None
    doc_type: str | None = None
    publisher: str | None = None


@dataclass(frozen=True)
class QaContext:
    """问答用到的外部能力。

    集中成一个对象，理由同 IndexingContext：同一组依赖要传给好几个步骤，
    散着传容易漏掉某个或把两个参数传反。
    """

    mysql: MysqlStore
    milvus: MilvusStore
    embedding: EmbeddingStore
    client: BailianClient
    redis: RedisStore
    config: RetrievalConfig


@dataclass(frozen=True)
class QaResponse:
    """一次问答的完整结果。

    answer.text 是最终要展示的文本，拒答或降级时已被替换过，调用方直接
    展示即可。

    只带 top_score 与 parent_count，不带召回到的父块全文：父块由界面按
    parent_id 单独拉取（见 04-api-contract 2.3）。

    refused 与 answer.degraded 是两件事——前者是检索不够、根本没调模型，
    后者是调了但回答没有可核对的出处。排查方向不同。
    """

    trace_id: str
    session_id: str
    answer: Answer
    top_score: float | None
    parent_count: int
    refused: bool
    cache_hit: bool = False
    notes: list[str] = field(default_factory=list)


def _cache_digest(request: QaRequest) -> str:
    """缓存键：把影响回答的输入拼起来取哈希。

    用哈希而不是原文：问题可能有几千字符，直接做键会很长；问题里还可能
    带换行等字符。

    三个过滤维度都要进来——同一个问题在不同筛选下召回的材料不同，答案
    也可能不同。
    """
    payload = json.dumps(
        [request.question, request.country, request.doc_type, request.publisher],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_query(request: QaRequest) -> RetrievalQuery:
    """把用例请求转成检索请求。"""
    return RetrievalQuery(
        text=request.question,
        country=request.country,
        doc_type=request.doc_type,
        publisher=request.publisher,
    )


def _encode_child(child: RetrievedChild) -> dict[str, Any]:
    return {
        "chunk_id": child.chunk_id,
        "score": child.score,
        "text": child.text,
        "start_offset": child.start_offset,
        "end_offset": child.end_offset,
    }


def _decode_child(payload: dict[str, Any]) -> RetrievedChild:
    return RetrievedChild(
        chunk_id=payload["chunk_id"],
        score=payload["score"],
        text=payload["text"],
        start_offset=payload["start_offset"],
        end_offset=payload["end_offset"],
    )


def _encode_parent(parent: RetrievedParent) -> dict[str, Any]:
    return {
        "parent_id": parent.parent_id,
        "doc_id": parent.doc_id,
        "score": parent.score,
        "text": parent.text,
        "token_count": parent.token_count,
        "title": parent.title,
        "source_url": parent.source_url,
        "country": parent.country,
        "doc_type": parent.doc_type,
        "publisher": parent.publisher,
        "effective_date": (
            parent.effective_date.isoformat() if parent.effective_date else None
        ),
        "matched_children": [
            _encode_child(child) for child in parent.matched_children
        ],
    }


def _decode_parent(payload: dict[str, Any]) -> RetrievedParent:
    raw_date = payload["effective_date"]
    return RetrievedParent(
        parent_id=payload["parent_id"],
        doc_id=payload["doc_id"],
        score=payload["score"],
        text=payload["text"],
        token_count=payload["token_count"],
        title=payload["title"],
        source_url=payload["source_url"],
        country=payload["country"],
        doc_type=payload["doc_type"],
        publisher=payload["publisher"],
        effective_date=date.fromisoformat(raw_date) if raw_date else None,
        matched_children=[
            _decode_child(child) for child in payload["matched_children"]
        ],
    )


def _encode(response: QaResponse) -> dict[str, Any]:
    """把响应压成可 JSON 化的结构，供缓存。

    带上被引用的材料全文：缓存命中的请求不会再检索一次，材料得从缓存里
    拿得出来，否则引用只剩一个 id，界面点开是空的。
    """
    return {
        "text": response.answer.text,
        "raw_text": response.answer.raw_text,
        "degraded": response.answer.degraded,
        "notes": list(response.answer.notes),
        "cited": [_encode_parent(parent) for parent in response.answer.citations.cited],
        "invalid_numbers": list(response.answer.citations.invalid_numbers),
        "top_score": response.top_score,
        "parent_count": response.parent_count,
        "refused": response.refused,
    }


def _decode(
    payload: dict[str, Any], *, trace_id: str, session_id: str
) -> QaResponse:
    """从缓存内容重建响应。"""
    return QaResponse(
        trace_id=trace_id,
        session_id=session_id,
        answer=Answer(
            text=payload["text"],
            raw_text=payload["raw_text"],
            citations=CitationReport(
                cited=[_decode_parent(item) for item in payload["cited"]],
                invalid_numbers=list(payload["invalid_numbers"]),
            ),
            degraded=payload["degraded"],
            notes=list(payload["notes"]),
        ),
        top_score=payload["top_score"],
        parent_count=payload["parent_count"],
        refused=payload["refused"],
        cache_hit=True,
    )


def _remember(
    request: QaRequest, response: QaResponse, context: QaContext
) -> None:
    """把这一问一答写回会话与缓存。

    **拒答的也写会话**：使用者接着问「那到底是多少」时，前端与服务端都该
    知道上一轮答的是什么，否则上下文断在半路。

    缓存只在单轮时写，理由见 answer()。
    """
    if request.session_id is not None:
        context.redis.append_message(
            response.session_id, {"role": ROLE_USER, "content": request.question}
        )
        context.redis.append_message(
            response.session_id,
            {"role": ROLE_ASSISTANT, "content": response.answer.text},
        )
        return

    context.redis.cache_answer(_cache_digest(request), _encode(response))


def answer(
    request: QaRequest, context: QaContext, *, trace_id: str | None = None
) -> QaResponse:
    """跑完一次问答。

    **只有单轮请求走缓存。** 多轮的回答依赖前面的对话内容，同一个问题在
    不同上下文里答案可能不同；用同一个键缓存会串。

    **会话上下文不参与提示词。** 首期的多轮只做到「记下来」，检索与生成
    仍只看当前这一问。把它接进提示词是另一轮的事——那要先想清楚历史轮次
    里的材料算不算引用来源，而参考来源目前必须全部来自本轮检索。
    """
    trace = trace_id if trace_id is not None else str(uuid.uuid4())
    session_id = request.session_id or str(uuid.uuid4())
    single_turn = request.session_id is None

    if single_turn:
        cached = context.redis.get_cached_answer(_cache_digest(request))
        if cached is not None:
            return _decode(cached, trace_id=trace, session_id=session_id)

    result = answer_question(
        _to_query(request),
        config=context.config,
        mysql=context.mysql,
        milvus=context.milvus,
        embedding=context.embedding,
        client=context.client,
    )

    response = QaResponse(
        trace_id=trace,
        session_id=session_id,
        answer=result.answer,
        top_score=result.retrieval.top_score,
        parent_count=len(result.retrieval.parents),
        refused=result.refused,
    )

    _remember(request, response, context)
    return response
