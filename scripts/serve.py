"""启动 HTTP 服务。

用法：
    python scripts/serve.py

起来之后：
    接口文档  http://127.0.0.1:8000/docs
    健康检查  http://127.0.0.1:8000/api/v1/health

冷启动会先加载嵌入模型（约 10 秒），之后常驻显存，不再重复加载。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import uvicorn  # noqa: E402

from cbe_rag.api import create_app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


def main() -> int:
    # 只监听本机：首期没有鉴权，绑到 0.0.0.0 等于把接口暴露给整个内网
    uvicorn.run(create_app(), host=HOST, port=PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
