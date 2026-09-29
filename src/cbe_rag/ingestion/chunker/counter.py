"""token 计数。

切分按 token 数而不是字符数：bge-m3 用 XLM-RoBERTa 的词表，
中英文的字符-token 比差别很大。实测：

    英文  67 字符 → 19 token（约 3.5 字符/token）
    中文  24 字符 → 14 token（约 1.7 字符/token）

同样长度的中文文本，token 数差不多是英文的两倍。按字符数估算会让
中文块偏大、英文块偏小，而 token 数才决定能不能喂进模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class TokenizerProtocol(Protocol):
    """tokenizer 需要提供的能力。

    定义成协议是为了让单元测试注入假实现——`models/` 目录不进版本库，
    靠真实词表跑测试会让换台机器就挂。
    """

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        ...


class TokenCountError(Exception):
    """token 计数无法完成时抛出。"""


class TokenCounter:
    """文本 token 计数器。

    只加载词表，不加载模型权重：实测约 1 秒，且不占显存。
    与嵌入用同一个词表，避免两边对不齐。
    """

    def __init__(
        self,
        model_path: Path,
        tokenizer: TokenizerProtocol | None = None,
    ) -> None:
        """初始化。

        tokenizer 仅用于测试注入；生产路径下按 model_path 加载词表。
        """
        self._tokenizer: TokenizerProtocol = (
            tokenizer if tokenizer is not None else self._load(model_path)
        )

    @staticmethod
    def _load(model_path: Path) -> TokenizerProtocol:
        """按模型路径加载词表。

        导入放在函数内：transformers 会连带拉起不少东西，
        放在模块顶部会让 import 变慢，而多数场景用不到计数。
        """
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_path))

    def count(self, text: str) -> int:
        """返回文本的 token 数。

        失败时抛出 TokenCountError 而不是返回一个估计值——
        错误的 token 数会让切分结果悄悄跑偏，比直接报错难排查得多。
        """
        try:
            return len(self._tokenizer.encode(text))
        except Exception as exc:
            raise TokenCountError(
                "计算 token 数失败：%s: %s" % (type(exc).__name__, exc)
            ) from exc

    def count_all(self, texts: list[str]) -> list[int]:
        """批量计数，返回顺序与输入一致。"""
        return [self.count(text) for text in texts]
