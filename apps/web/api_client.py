"""后端接口的访问封装。

界面只跟这一层打交道，不直接拼 URL 或处理响应结构——接口地址要改，
或者响应形状要跟着契约变，都只动这里。

用标准库的 urllib 而不是引入 httpx：这一层只有四个调用，不值得为它
加一个依赖。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 180.0


class ApiError(Exception):
    """后端返回了非 2xx，或压根连不上。

    带上 code 与状态码：界面要据此说清是「后端没起」还是「这个请求
    本身有问题」，两者的处理完全不同。
    """

    def __init__(self, message: str, *, code: str = "", status: int = 0) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


class BackendClient:
    """后端接口的客户端。"""

    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, timeout: float = DEFAULT_TIMEOUT
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _request(
        self, path: str, *, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """发一次请求并解出 JSON。

        **连不上与业务错误分开报。** 前者是后端没起，后者是请求本身的问题，
        界面上的提示不一样——混成一句「请求失败」，用的人不知道该去启后端
        还是改自己的输入。
        """
        url = "%s%s" % (self._base_url, path)
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"

        request = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            message, code = _describe_http_error(exc)
            raise ApiError(message, code=code, status=exc.code) from exc
        except urllib.error.URLError as exc:
            raise ApiError(
                "连不上后端（%s）。确认服务已启动：python scripts/serve.py"
                % self._base_url
            ) from exc

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """提一个问题。"""
        payload: dict[str, Any] = {"question": question}
        if session_id is not None:
            payload["session_id"] = session_id
        if filters:
            payload["filters"] = filters
        return self._request("/api/v1/ask", payload=payload)

    def get_parent(self, parent_id: str) -> dict[str, Any]:
        """按 parent_id 取父块全文。"""
        return self._request("/api/v1/parents/%s" % parent_id)

    def health(self) -> dict[str, Any]:
        """取各依赖的连通性。"""
        return self._request("/api/v1/health")


def _describe_http_error(exc: urllib.error.HTTPError) -> tuple[str, str]:
    """从错误响应里取出契约约定的 code 与 message。

    后端按契约返回统一形状；万一拿到的不是那个形状（比如反向代理插了一
    层），退回原始文本——总比只报一个状态码强。
    """
    try:
        body = json.loads(exc.read().decode("utf-8"))
        error = body.get("error") or {}
        code = str(error.get("code") or "")
        message = str(error.get("message") or exc.reason)
        return message, code
    except Exception:
        return "请求失败（HTTP %d）" % exc.code, ""
