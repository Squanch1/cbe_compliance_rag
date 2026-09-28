"""健康检查结果的统一结构。

四个存储适配器的 health_check() 都返回这个结构，
连通性脚本与健康检查接口据此统一渲染，不必各自处理返回格式。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthResult:
    """单个外部服务的健康检查结果。

    detail 字段在成功时放版本号等有用信息，失败时放异常类型与消息。
    脚本直接把它打印给使用者，因此这里的内容要能让人看懂发生了什么。
    """

    service: str
    ok: bool
    detail: str
    elapsed_ms: float

    def render(self) -> str:
        """渲染成一行可读文本，供连通性脚本输出。"""
        status = "OK  " if self.ok else "FAIL"
        return "[%s] %-8s %8.1fms  %s" % (status, self.service, self.elapsed_ms, self.detail)
