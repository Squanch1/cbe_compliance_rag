"""接口层：FastAPI 路由与请求响应模型。

只依赖 services/ 与 config/，业务规则不写在这里（见 02-architecture 第 7 节）。
"""

from cbe_rag.api.app import create_app
from cbe_rag.api.deps import ApiError, AppState, build_state
from cbe_rag.api.schemas import (
    AskRequest,
    AskResponse,
    ErrorResponse,
    HealthResponse,
    ParentOut,
)

__all__ = [
    "ApiError",
    "AppState",
    "AskRequest",
    "AskResponse",
    "ErrorResponse",
    "HealthResponse",
    "ParentOut",
    "build_state",
    "create_app",
]
