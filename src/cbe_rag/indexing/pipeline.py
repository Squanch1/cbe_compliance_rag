"""索引流程。

把判重、解析、切块、向量化、落库串起来。

判重单独成一步（`prepare_document`），因为它只依赖 MySQL 与文件本身，
不碰解析和嵌入，可以脱离模型与 Milvus 来测——而判重恰恰是错得最安静的
一环：判错的后果是同一份文档重复入库，或者文档悄悄不再被检索到，两种
都不报错。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from cbe_rag.ingestion.fetcher.collector import CollectedDocument
from cbe_rag.ingestion.parser.schema import new_doc_id
from cbe_rag.indexing.decision import decide
from cbe_rag.indexing.hashing import file_content_hash
from cbe_rag.indexing.models import Decision, ImportAction
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.mysql_store import MysqlStore
from cbe_rag.storage.records import DocumentRecord


@dataclass(frozen=True)
class PreparedDocument:
    """判重准备的结果。

    content_hash 一并带出来：构造 documents 记录时要写它，重算一遍等于
    把同一个文件读两次。
    """

    document: CollectedDocument
    content_hash: str
    decision: Decision


def prepare_document(
    document: CollectedDocument, mysql: MysqlStore
) -> PreparedDocument:
    """算哈希、查库、判定这份文档该怎么处理。

    **只查不写。** 判定结果由调用方执行，判定过程本身不会在库里留下
    任何痕迹，因此这一步失败或中断都不影响数据。

    按 URL 查只在两个条件同时满足时做：哈希没命中（命中说明文件一个
    字节都没变，同链接下还有没有别的记录不影响本次处理），且清单登记
    了 source_url（没登记就无从查起）。省下的是一次查询，也少一条
    「拿 None 去查」的岔路。
    """
    content_hash = file_content_hash(document.raw_path)
    by_hash = mysql.get_document_by_hash(content_hash)

    source_url = document.meta.source_url
    by_url = (
        mysql.find_active_by_source_url(source_url)
        if by_hash is None and source_url is not None
        else None
    )

    return PreparedDocument(
        document=document,
        content_hash=content_hash,
        decision=decide(document.meta, by_hash=by_hash, by_url=by_url),
    )


def _to_record(
    document: CollectedDocument,
    content_hash: str,
    doc_id: str,
    status: DocumentStatus,
) -> DocumentRecord:
    """把采集结果与判重结果拼成一条 documents 记录。"""
    meta = document.meta
    return DocumentRecord(
        doc_id=doc_id,
        content_hash=content_hash,
        status=status,
        title=meta.title,
        platform=meta.platform,
        source_url=meta.source_url,
        publisher=meta.publisher,
        country=meta.country,
        doc_type=meta.doc_type,
        effective_date=meta.effective_date,
        collected_date=meta.collected_date,
        raw_path=str(document.raw_path),
        missing_fields=tuple(meta.missing_required_fields()),
    )


def register_document(
    document: CollectedDocument,
    prepared: PreparedDocument,
    mysql: MysqlStore,
    *,
    now: datetime | None = None,
) -> DocumentRecord:
    """把这份文档登记到 documents 表，返回写进去的那条记录。

    - `NEW` 与 `SUPERSEDE_AND_NEW`：新建记录，生成新的 doc_id
    - `REINDEX`：复用命中记录的 doc_id，更新元数据并把状态拨回 pending

    **登记发生在解析之前。** 元数据不齐的文档也要留下记录，才能查到
    「还差哪些没补齐」；解析失败或中途出错时，pending 也标出了这是一份
    没跑完的文档，比库里什么都没有强。

    **doc_id 在判重通过后才生成。** 命中时必须复用库里那个——chunk_id
    由它派生（`{doc_id}_pXXXX`），换一个就等于在向量库里另起一套前缀，
    同一份文档会出现两组向量，而旧的那组还删不掉。
    """
    decision = prepared.decision

    if decision.action is ImportAction.REINDEX:
        matched = decision.matched
        if matched is None:
            raise ValueError("REINDEX 缺少命中记录，判重结果不完整")
        record = _to_record(
            document, prepared.content_hash, matched.doc_id, DocumentStatus.PENDING
        )
        mysql.update_document_meta(record, now=now)
        mysql.update_document_status(
            matched.doc_id, DocumentStatus.PENDING, now=now
        )
        return record

    record = _to_record(
        document,
        prepared.content_hash,
        new_doc_id(),
        DocumentStatus.PENDING,
    )
    mysql.insert_document(record, now=now)
    return record
