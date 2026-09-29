"""解析层级的接口与结果类型。

多层路由把「同一份文件换不同工具尝试解析」抽象成一条链：链上每一层
是一种解析器，从便宜到贵排列，前一层不合格就换下一层，全部不合格
再交人工。

本模块只定义类型，不含逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cbe_rag.ingestion.parser.quality import QualityReport
from cbe_rag.ingestion.parser.schema import ParsedDocument


@dataclass(frozen=True)
class ParseRequest:
    """一次解析请求。

    **解析层不认识 fetcher 的 CollectedDocument**，用自己定义的输入类型，
    否则解析层就反向依赖了上层。调用方负责把采集结果转成这个结构。
    """

    doc_id: str
    title: str
    source_url: str | None
    raw_path: Path


class TierParser(Protocol):
    """链上的一层。

    实现者只需保证「输入 ParseRequest，输出 ParsedDocument」；
    失败时抛异常即可，换层与兜底的逻辑由路由负责。
    """

    name: str
    """层级标识，用于日志与人工待办记录，如 "pdf.text_layer"。"""

    def parse(self, request: ParseRequest) -> ParsedDocument:
        """解析。失败时抛异常。"""
        ...


@dataclass(frozen=True)
class TierAttempt:
    """链上某一层的尝试记录。

    ok 为 False 时 detail 必须写清原因——人工处理时最想知道的是
    「机器试过什么、卡在哪」。只记一句「解析失败」等于没记，
    人工得从头再来一遍。
    """

    tier: str
    ok: bool
    detail: str
    report: QualityReport | None = None


@dataclass(frozen=True)
class ParseOutcome:
    """整条链跑完的结果。

    document 为 None 表示所有层都不合格，需人工处理；
    attempts 保留每一层的尝试记录，供人工排查。
    """

    document: ParsedDocument | None
    attempts: list[TierAttempt]
    difficulty: str = ""

    @property
    def needs_manual(self) -> bool:
        """是否需要人工处理。"""
        return self.document is None
