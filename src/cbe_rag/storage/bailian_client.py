"""阿里云百炼的调用适配器。

走 OpenAI 兼容模式（见 CLAUDE.md 第 3 节）。业务层通过本类调用大模型，
不直接 import openai。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from cbe_rag.config.settings import LlmConfig
from cbe_rag.storage.health import HealthResult


@dataclass(frozen=True)
class Completion:
    """一次调用的产出。

    用量的两个字段允许为空：兼容接口不保证返回 usage，而它只是观测数据，
    缺了不该让一次成功的调用变成失败。
    """

    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BailianClient:
    """百炼访问入口。

    构造本对象不发起网络请求；客户端在首次调用时建立并复用。
    """

    def __init__(
        self,
        config: LlmConfig,
        create_factory: Callable[[], Callable[..., Any]] | None = None,
    ) -> None:
        """初始化。

        create_factory 仅用于测试注入，它返回的应当是
        `chat.completions.create` 那一层的可调用对象。
        """
        self._config = config
        self._create_factory: Callable[[], Callable[..., Any]] = (
            create_factory if create_factory is not None else self._load_create
        )
        self._create: Callable[..., Any] | None = None

    def _load_create(self) -> Callable[..., Any]:
        """建立客户端并返回调用入口。

        导入放在函数内：把 SDK 的导入推迟到真正要用的时候，
        `import cbe_rag.storage` 不必为它付出代价。
        """
        from openai import OpenAI

        client = OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key.get_secret_value(),
            timeout=self._config.timeout_seconds,
        )
        return client.chat.completions.create

    def _get_create(self) -> Callable[..., Any]:
        """返回调用入口，没有就先建一个。"""
        if self._create is None:
            self._create = self._create_factory()
        return self._create

    def complete(self, system: str, user: str) -> Completion:
        """按系统与用户消息调用一次，返回文本与用量。

        **不吞异常。** 调用失败必须让调用方知道：拿一个空回答继续往下走，
        在界面上与「依据不足」看不出区别，而两者的处理方式完全不同。
        """
        response = self._get_create()(
            model=self._config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )
        usage = getattr(response, "usage", None)
        return Completion(
            text=response.choices[0].message.content or "",
            model=self._config.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

    def health_check(self) -> HealthResult:
        """检查配置是否齐备。

        **不发起真实调用**：连通性检查可能被频繁执行，每次都烧一点额度
        不合适。真正的可用性由首次 complete 验证。
        """
        started = time.perf_counter()
        detail = "base_url=%s model=%s temperature=%.1f max_tokens=%d" % (
            self._config.base_url,
            self._config.model,
            self._config.temperature,
            self._config.max_tokens,
        )
        if self._create is not None:
            detail += " 客户端=已建立"

        return HealthResult(
            service="bailian",
            ok=True,
            detail=detail,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def close(self) -> None:
        """释放调用入口。

        每次调用自行复用同一个客户端，这里只丢掉引用，不额外做关闭动作
        ——openai 的客户端没有需要显式释放的连接。
        """
        self._create = None
