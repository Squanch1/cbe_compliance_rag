"""从存储读出、或将要写入存储的业务记录。

与 ddl.py 的分工：ddl.py 回答「表长什么样」，本模块回答「一行是什么」。

定义在 storage 层而不是 indexing 层，是因为它由适配器返回、也由适配器
接收——适配器不该反过来依赖调用方。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from cbe_rag.storage.ddl import DocumentStatus


@dataclass(frozen=True)
class DocumentRecord:
    """`documents` 表里的一行。

    读与写共用同一个类型：两者字段完全一致，分成两个再互相转换，
    除了多几处赋值没有别的作用。

    **不直接嵌入 `DocumentMeta`**：那样 storage 层就要 import 解析层的
    类型，而依赖方向是解析 -> 索引 -> 存储。字段平铺虽然多写几行，但
    保住了方向，也让「表列与字段一一对应」这件事一眼可见。
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
    raw_path: str
    # 元数据齐备时为空元组，这也是最常见的情况
    missing_fields: tuple[str, ...] = ()
