"""用例编排。

把领域层的模块串成一次完整的用例：取会话、查缓存、检索、生成、写回。
接口层只做参数校验与格式转换，业务规则都在这里。
"""

from cbe_rag.services.qa import QaContext, QaRequest, QaResponse, answer

__all__ = [
    "QaContext",
    "QaRequest",
    "QaResponse",
    "answer",
]
