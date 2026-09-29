"""按 parent_id 恢复父块，并把引用元数据配齐。

Milvus 里只有子块的向量和几个标量字段；父块正文与引用元数据都在 MySQL。
这一步按 parent_id 把父块正文取回来，按 doc_id 把元数据取回来，拼成可以
直接送进提示词的 `RetrievedParent`。
"""

from __future__ import annotations

from cbe_rag.retrieval.models import ParentHit, RetrievedParent
from cbe_rag.storage.mysql_store import MysqlStore


class ContextError(Exception):
    """上下文组装无法继续时抛出。"""


def load_parents(
    hits: list[ParentHit], mysql: MysqlStore
) -> list[RetrievedParent]:
    """按 parent_id 取父块正文、按 doc_id 取引用元数据。

    返回顺序与传进来的 hits 一致，也就是相关性从高到低——调用方拿到的
    上下文顺序就是它该有的顺序，不必再排一次。

    两个查询都是批量的：保留 5 个父块，逐条查就是 10 次往返，而它们本来
    就是两批。

    **父块或文档记录缺失时抛错，不跳过。** Milvus 里有子块指向它、MySQL
    里却没有，说明索引中途断了（向量先写、块后写，中间失败）。少一条
    上下文不会报错，只会让答案缺一块，反而更难发现。
    """
    if not hits:
        return []

    parent_ids = [hit.parent_id for hit in hits]
    chunks = {chunk.chunk_id: chunk for chunk in mysql.get_chunks(parent_ids)}

    missing_chunks = [
        parent_id for parent_id in parent_ids if parent_id not in chunks
    ]
    if missing_chunks:
        raise ContextError(
            "有 %d 个父块在库里找不到：%s。Milvus 里有指向它们的子块，"
            "说明索引时中途断了。"
            % (len(missing_chunks), "、".join(missing_chunks[:5]))
        )

    # 去重但保序：多个父块常常属于同一份文档，同一份只查一次
    doc_ids = list(dict.fromkeys(hit.doc_id for hit in hits))
    documents = {record.doc_id: record for record in mysql.get_documents(doc_ids)}

    missing_docs = [doc_id for doc_id in doc_ids if doc_id not in documents]
    if missing_docs:
        raise ContextError(
            "有 %d 份文档的元数据找不到：%s。引用要带出处与生效日期，"
            "缺了就没法生成合规的引用。"
            % (len(missing_docs), "、".join(missing_docs[:5]))
        )

    return [
        RetrievedParent(
            parent_id=hit.parent_id,
            doc_id=hit.doc_id,
            score=hit.score,
            text=chunks[hit.parent_id].text,
            title=documents[hit.doc_id].title,
            source_url=documents[hit.doc_id].source_url,
            country=documents[hit.doc_id].country,
            doc_type=documents[hit.doc_id].doc_type,
            publisher=documents[hit.doc_id].publisher,
            effective_date=documents[hit.doc_id].effective_date,
        )
        for hit in hits
    ]
