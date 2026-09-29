"""百炼客户端的单元测试。

注入假的调用入口，不发起真实请求。

关心两件事：传下去的参数对不对（模型名、温度、token 上限），以及
异常有没有被吞掉。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from cbe_rag.config.settings import LlmConfig
from cbe_rag.storage.bailian_client import BailianClient


def make_config(**overrides: Any) -> LlmConfig:
    fields: dict[str, Any] = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": SecretStr("fake-key"),
        "model": "qwen-plus",
        "timeout_seconds": 30.0,
        "max_tokens": 2048,
        "temperature": 0.0,
    }
    fields.update(overrides)
    return LlmConfig(**fields)


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class FakeResponse:
    def __init__(self, content: str | None, usage: FakeUsage | None) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = usage


class FakeCreate:
    """假的调用入口，记录收到的参数。"""

    def __init__(
        self,
        content: str | None = "回答",
        usage: FakeUsage | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self._content = content
        self._usage = usage
        self._fail_with = fail_with
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeResponse:
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append(kwargs)
        return FakeResponse(self._content, self._usage)


def build_client(**create_kwargs: Any) -> tuple[BailianClient, FakeCreate]:
    """构造客户端并返回它使用的假调用入口。"""
    create = FakeCreate(**create_kwargs)
    client = BailianClient(make_config(), create_factory=lambda: create)
    return client, create


class TestComplete:
    def test_returns_the_text(self) -> None:
        client, _ = build_client(content="进口一站式服务适用于 150 欧元以下。")

        result = client.complete("系统", "用户")

        assert result.text == "进口一站式服务适用于 150 欧元以下。"

    def test_carries_the_model_name(self) -> None:
        client, _ = build_client()

        assert client.complete("系统", "用户").model == "qwen-plus"

    def test_sends_both_messages(self) -> None:
        client, create = build_client()

        client.complete("这是系统提示词", "这是用户消息")

        messages = create.calls[0]["messages"]
        assert messages[0] == {"role": "system", "content": "这是系统提示词"}
        assert messages[1] == {"role": "user", "content": "这是用户消息"}

    def test_applies_the_configured_limits(self) -> None:
        client, create = build_client()

        client.complete("系统", "用户")
        call = create.calls[0]

        assert call["model"] == "qwen-plus"
        assert call["max_tokens"] == 2048
        assert call["temperature"] == 0.0

    def test_records_usage_when_returned(self) -> None:
        client, _ = build_client(usage=FakeUsage(prompt=1200, completion=180))

        result = client.complete("系统", "用户")

        assert result.prompt_tokens == 1200
        assert result.completion_tokens == 180

    def test_usage_may_be_absent(self) -> None:
        # 兼容接口不保证返回用量，而它只是观测数据，
        # 缺了不该让一次成功的调用变成失败
        client, _ = build_client(usage=None)

        result = client.complete("系统", "用户")

        assert result.text == "回答"
        assert result.prompt_tokens is None

    def test_empty_content_becomes_empty_string(self) -> None:
        # content 可能是 None（模型只返回了 tool_calls 之类），
        # 转成空串让调用方按「没有内容」处理，而不是崩在 None 上
        client, _ = build_client(content=None)

        assert client.complete("系统", "用户").text == ""

    def test_reuses_the_same_entry_point(self) -> None:
        # 每次调用重新建客户端是白付一次握手成本
        created: list[FakeCreate] = []

        def factory() -> FakeCreate:
            create = FakeCreate()
            created.append(create)
            return create

        client = BailianClient(make_config(), create_factory=factory)
        client.complete("系统", "用户")
        client.complete("系统", "用户")

        assert len(created) == 1


class TestFailure:
    def test_call_failure_is_raised_not_swallowed(self) -> None:
        # 吞掉异常会让调用方拿一个空回答继续往下走，
        # 而「空回答」与「依据不足」在界面上看不出区别
        client, _ = build_client(fail_with=RuntimeError("rate limited"))

        with pytest.raises(RuntimeError, match="rate limited"):
            client.complete("系统", "用户")


class TestLifecycle:
    def test_health_check_reports_the_configuration(self) -> None:
        client, _ = build_client()

        result = client.health_check()

        assert result.ok is True
        assert result.service == "bailian"
        assert "qwen-plus" in result.detail

    def test_health_check_does_not_call_the_model(self) -> None:
        # 连通性检查可能被频繁执行，每次烧一点额度不合适
        client, create = build_client()

        client.health_check()

        assert create.calls == []

    def test_close_drops_the_entry_point(self) -> None:
        client, _ = build_client()
        client.complete("系统", "用户")

        client.close()

        assert client.health_check().detail.find("已建立") == -1
