"""日志与链路追踪。

可被任意层依赖，本层不反向依赖任何其他层
（见 docs/spec/02-architecture.md 第 7 节）。
"""

from cbe_rag.observability.logging import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
