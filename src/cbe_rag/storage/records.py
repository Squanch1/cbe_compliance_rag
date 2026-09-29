"""从存储读出来的业务记录。

与 ddl.py 的分工：ddl.py 回答「表长什么样」，本模块回答「读出来的一行
是什么」。

定义在 storage 层而不是 indexing 层，是因为它由适配器返回——适配器不该
反过来依赖调用方。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from cbe_rag.storage.ddl import DocumentStatus


@dataclass(frozen=True)
class ExistingDocument:
    """`documents` 表里的一行。

    字段与表列一一对应，判重时用来与清单上登记的元数据比对。

    **不直接嵌入 `DocumentMeta`**：那样 storage 层就要 import 解析层的
    类型，而两边本是单向依赖（解析 -> 索引 -> 存储）。平铺字段虽然多写
    几行，但保住了方向，也让「表列与字段一一对应」这件事一眼可见。
    """

    doc_id: str
    content_hash: str
    status: DocumentStatus
    title: str
    platform: str
    source_url: str | None
    publisher: str | None
    country: str | None
    doc_type: str | None
    effective_date: date | None
    collected_date: date
