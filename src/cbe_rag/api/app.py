"""FastAPI 应用。

启动时建好各适配器、关闭时释放；装上 trace_id 中间件与统一的错误处理器。
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cbe_rag.api.deps import ApiError, build_state
from cbe_rag.api.routes import router
from cbe_rag.api.schemas import ErrorBody, ErrorResponse
from cbe_rag.config.settings import Settings
from cbe_rag.retrieval.service import UncalibratedThresholdError

# 调用方自带 trace_id 时用的请求头。
TRACE_HEADER = "X-Trace-Id"

# trace_id 的形状限制。它要进日志与响应，允许任意字符会让日志格式被破坏。
_TRACE_MAX_LENGTH = 128
_TRACE_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)


def is_valid_trace(value: str) -> bool:
    """判断调用方给的 trace_id 能不能直接用。

    不合规就重新生成，而不是报错：这个值只是排查用的线索，为它让整个
    请求失败不值当。
    """
    return (
        bool(value)
        and len(value) <= _TRACE_MAX_LENGTH
        and all(char in _TRACE_ALLOWED for char in value)
    )


def _error_response(
    request: Request, status_code: int, code: str, message: str
) -> JSONResponse:
    """按契约里的统一形状构造错误响应。"""
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            trace_id=getattr(request.state, "trace_id", ""),
            error=ErrorBody(code=code, message=message),
        ).model_dump(mode="json"),
    )


def _register_handlers(app: FastAPI) -> None:
    """把各类异常映射成契约里约定的错误码。"""

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(UncalibratedThresholdError)
    async def handle_uncalibrated(
        request: Request, exc: UncalibratedThresholdError
    ) -> JSONResponse:
        # 单独成一类而不并入 internal_error：它不是缺陷，是刻意不让服务在
        # 未校准的状态下放行结果。运维看到该去标定阈值，不是查 bug。
        return _error_response(request, 503, "uncalibrated_threshold", str(exc))

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(request, 400, "invalid_request", str(exc))


def create_app(
    settings: Settings | None = None,
    state_factory: Callable[[Settings | None], object] | None = None,
) -> FastAPI:
    """构造应用。

    settings 与 state_factory 可注入，便于测试起一个不连真实服务的实例。
    """

    factory = state_factory or build_state
    # 构造时就建好状态，而不是等 lifespan：TestClient 只在 with 块里才跑
    # lifespan，状态放在那里会让没进 with 的调用一律拿到空属性。
    # 各适配器构造时不产生网络调用（见 deps.build_state），提前建没有代价。
    app_state = factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        app_state.close()

    app = FastAPI(
        title="跨境电商合规问答",
        version="0.1.0",
        description="平台规则与税务合规问答，所有回答附带可追溯的原文引用。",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_trace_id(
        request: Request, call_next: Callable[[Request], object]
    ) -> object:
        """给每个请求定一个 trace_id，并回写到响应头。

        调用方自带且格式合规时沿用——跨服务串联时用得上。
        """
        supplied = request.headers.get(TRACE_HEADER, "")
        request.state.trace_id = (
            supplied if is_valid_trace(supplied) else str(uuid.uuid4())
        )
        response = await call_next(request)
        response.headers[TRACE_HEADER] = request.state.trace_id
        return response

    app.state.app_state = app_state
    _register_handlers(app)
    app.include_router(router)
    return app
