"""接口的请求与响应模型。

形状以 docs/spec/04-api-contract.md 为准，改这里之前先改那份文档。

**这一层只做格式转换**：把用例层的返回摊成 JSON 友好的结构，把 Pydantic
的校验错误映射成契约里约定的错误码。业务规则不写在这里。
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from cbe_rag.services.qa import QaResponse


class Filters(BaseModel):
    """检索的过滤维度。三个都可选，不传即不筛。"""

    model_config = ConfigDict(extra="forbid")

    country: str | None = Field(default=None, max_length=16)
    doc_type: str | None = Field(default=None, max_length=32)
    publisher: str | None = Field(default=None, max_length=64)


class AskRequest(BaseModel):
    """一次问答请求。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=128)
    filters: Filters | None = None


class MatchedChildOut(BaseModel):
    """一条命中的子块。

    start_offset / end_offset 指向**父块正文里的位置**，界面按它们高亮。
    """

    chunk_id: str
    text: str
    start_offset: int | None
    end_offset: int | None
    score: float


class CitationOut(BaseModel):
    """一条引用：一份被引用的材料，连同其中命中的段落。"""

    parent_id: str
    title: str
    source_url: str | None
    effective_date: date | None
    country: str | None
    doc_type: str | None
    publisher: str | None
    score: float
    matched_children: list[MatchedChildOut]


class RetrievalOut(BaseModel):
    """检索的统计信息。

    不带召回到的父块全文：那由界面按 parent_id 单独拉（见契约 2.3）。
    """

    top_score: float | None
    parent_count: int


class AskResponse(BaseModel):
    """一次问答的响应。

    answer 是最终要展示的文本，拒答或降级时已被替换，客户端直接展示即可。

    refused 与 degraded 是两件事：前者是检索质量不够、根本没调模型，
    后者是调了但回答没有可核对的出处。排查方向不同。
    """

    trace_id: str
    session_id: str
    answer: str
    refused: bool
    degraded: bool
    cache_hit: bool
    citations: list[CitationOut]
    notes: list[str]
    retrieval: RetrievalOut


class ParentOut(BaseModel):
    """一条父块，供界面「展开上下文」时拉取。"""

    parent_id: str
    doc_id: str
    text: str
    token_count: int
    chunk_index: int
    title: str
    source_url: str | None
    effective_date: date | None
    collected_date: date
    country: str | None
    doc_type: str | None
    publisher: str | None


class DimensionOut(BaseModel):
    """维度表的一项。

    代码与中文名都给：界面显示中文，提交的却是代码——代码才是后端拼进
    检索表达式的东西。
    """

    code: str
    name_zh: str
    name_en: str


class DimensionsOut(BaseModel):
    """三张维度表的可用取值，供界面渲染下拉。

    只含启用中的取值。未启用的那些是给扩充范围预留的，摆到界面上只会让
    人选到一个筛不出东西的值。
    """

    countries: list[DimensionOut]
    doc_types: list[DimensionOut]
    publishers: list[DimensionOut]


class HealthItemOut(BaseModel):
    """单个依赖的连通性。"""

    service: str
    ok: bool
    detail: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    """健康检查响应。

    任一依赖不通时 ok 为 False，但 HTTP 状态仍是 200——这个接口的语义是
    「报告状态」，不是「服务本身能不能响应」。返回 503 会让监控把「某个
    依赖挂了」和「这个服务挂了」混为一谈。
    """

    ok: bool
    services: list[HealthItemOut]


class ErrorBody(BaseModel):
    """错误的机器可读标识与给人看的说明。

    客户端按 code 分支，不要按 message 匹配——文案会改，code 不会。
    """

    code: str
    message: str


class ErrorResponse(BaseModel):
    """所有非 2xx 响应共用的形状。"""

    trace_id: str
    error: ErrorBody


def to_ask_response(result: QaResponse) -> AskResponse:
    """把用例层的返回摊成响应模型。"""
    return AskResponse(
        trace_id=result.trace_id,
        session_id=result.session_id,
        answer=result.answer.text,
        refused=result.refused,
        degraded=result.answer.degraded,
        cache_hit=result.cache_hit,
        citations=[
            CitationOut(
                parent_id=parent.parent_id,
                title=parent.title,
                source_url=parent.source_url,
                effective_date=parent.effective_date,
                country=parent.country,
                doc_type=parent.doc_type,
                publisher=parent.publisher,
                score=parent.score,
                matched_children=[
                    MatchedChildOut(
                        chunk_id=child.chunk_id,
                        text=child.text,
                        start_offset=child.start_offset,
                        end_offset=child.end_offset,
                        score=child.score,
                    )
                    for child in parent.matched_children
                ],
            )
            for parent in result.answer.citations.cited
        ],
        notes=list(result.answer.notes),
        retrieval=RetrievalOut(
            top_score=result.top_score,
            parent_count=result.parent_count,
        ),
    )
