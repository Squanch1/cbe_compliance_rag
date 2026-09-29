"""token 计数器的单元测试。

注入假的 tokenizer，不加载真实的词表文件——
models/ 目录不进版本库，靠它跑测试会让换台机器就挂。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cbe_rag.ingestion.chunker.counter import TokenCounter, TokenCountError


class FakeTokenizer:
    """假的 tokenizer。

    用「每个字符一个 token」的最简规则，便于断言。
    """

    def __init__(self, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with
        self.calls: list[str] = []

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        if self._fail_with is not None:
            raise self._fail_with
        self.calls.append(text)
        return list(range(len(text)))


def build_counter(tokenizer: FakeTokenizer | None = None) -> tuple[TokenCounter, FakeTokenizer]:
    fake = tokenizer if tokenizer is not None else FakeTokenizer()
    return TokenCounter(Path("unused"), tokenizer=fake), fake


class TestCount:
    def test_returns_token_count(self) -> None:
        counter, _ = build_counter()

        assert counter.count("abcd") == 4

    def test_empty_text_counts_zero(self) -> None:
        counter, _ = build_counter()

        assert counter.count("") == 0

    def test_counts_are_independent(self) -> None:
        counter, _ = build_counter()

        assert counter.count("ab") == 2
        assert counter.count("abcd") == 4

    def test_delegates_to_injected_tokenizer(self) -> None:
        counter, fake = build_counter()

        counter.count("hello")

        assert fake.calls == ["hello"]

    def test_tokenizer_failure_is_wrapped(self) -> None:
        # 计数失败说明环境有问题，应明确报错而不是返回一个错误的数字
        # ——错误的 token 数会让切分结果悄悄跑偏
        counter, _ = build_counter(FakeTokenizer(fail_with=OSError("模型文件缺失")))

        with pytest.raises(TokenCountError, match="模型文件缺失"):
            counter.count("hello")


class TestBatch:
    def test_counts_a_batch(self) -> None:
        counter, fake = build_counter()

        assert counter.count_all(["ab", "cde"]) == [2, 3]

    def test_batch_preserves_order(self) -> None:
        counter, fake = build_counter()

        counts = counter.count_all(["a", "bbb", "cc"])

        assert counts == [1, 3, 2]
        assert fake.calls == ["a", "bbb", "cc"]

    def test_empty_batch(self) -> None:
        counter, _ = build_counter()

        assert counter.count_all([]) == []
