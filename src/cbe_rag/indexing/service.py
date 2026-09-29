"""批量导入。

把清单上的文档逐份交给索引流程，汇总成一份报告。

**一份文档失败不让整批停下。** 失败通常是个别文件的问题（文件头认不出、
坏文件），没有理由让剩下的几十份都不处理。失败记进报告，由人决定怎么
处置。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from cbe_rag.ingestion.fetcher.collector import (
    CollectionResult,
    CollectedDocument,
    collect_documents,
)
from cbe_rag.indexing.models import DocumentOutcome, ImportReport
from cbe_rag.indexing.pipeline import IndexingContext, index_document
from cbe_rag.storage.mysql_store import MysqlStore
from cbe_rag.storage.records import DocumentRecord


def _safe_index(
    document: CollectedDocument, context: IndexingContext
) -> DocumentOutcome:
    """处理一份文档，出错时记成失败而不是让整批停下。

    异常类型写进 detail：只说「失败」对排查没有帮助，而这里能提供的
    最有用的信息就是它是什么错。
    """
    try:
        return index_document(document, context)
    except Exception as exc:
        return DocumentOutcome(
            file_name=document.raw_path.name,
            # 异常可能发生在判重之前，这时确实还不知道该走哪条路
            action=None,
            ok=False,
            detail="%s: %s" % (type(exc).__name__, str(exc)[:200]),
        )


def _find_orphans(
    collection: CollectionResult, mysql: MysqlStore
) -> list[DocumentRecord]:
    """找出库里活跃、但清单里已经没有的文档。

    **只报告不动手。** 删除不可逆，而且清单改错一个字就会让文档从检索
    里消失，因此由人看过清单再决定。

    清单引用了但文件缺失的条目也要算作「清单里还有」：那一行还在，只是
    文件丢了，报成孤儿会误导人以为清单该删这一行。
    """
    known_paths = {str(document.raw_path) for document in collection.documents}
    known_paths |= {str(item.expected_path) for item in collection.missing_files}
    return [
        record
        for record in mysql.list_active_documents()
        if record.raw_path not in known_paths
    ]


def import_documents(
    manifest_path: Path,
    raw_dir: Path,
    context: IndexingContext,
    *,
    today: date | None = None,
) -> ImportReport:
    """按清单批量导入，返回汇总报告。

    today 透传给 collector，便于测试固定采集日期；生产路径下不传，
    取当天日期。
    """
    collection = collect_documents(manifest_path, raw_dir, today=today)

    return ImportReport(
        outcomes=[
            _safe_index(document, context) for document in collection.documents
        ],
        missing_files=collection.missing_files,
        orphans=_find_orphans(collection, context.mysql),
    )
