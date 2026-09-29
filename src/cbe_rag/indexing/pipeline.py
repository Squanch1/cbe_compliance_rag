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

from cbe_rag.ingestion.chunker.counter import TokenCounter
from cbe_rag.ingestion.chunker.service import chunk_document
from cbe_rag.ingestion.fetcher.collector import CollectedDocument
from cbe_rag.ingestion.parser.router import parse_document
from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel, new_doc_id
from cbe_rag.ingestion.parser.tier import ParseOutcome, ParseRequest
from cbe_rag.indexing.decision import decide
from cbe_rag.indexing.hashing import file_content_hash
from cbe_rag.indexing.models import Decision, DocumentOutcome, ImportAction
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.embedding import EmbeddingStore
from cbe_rag.storage.milvus_store import MilvusStore
from cbe_rag.storage.mysql_store import MysqlStore
from cbe_rag.storage.records import ChunkVector, DocumentRecord


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


def build_vectors(
    children: list[Chunk],
    record: DocumentRecord,
    embedding: EmbeddingStore,
) -> list[ChunkVector]:
    """把子块正文编码成向量库要的记录。

    **只编码子块。** 父块靠 parent_id 回 MySQL 取全文，不建向量
    （small-to-big）；存进 Milvus 只是白占空间。

    三个过滤维度从文档记录里冗余一份，好让检索直接在 Milvus 里按维度
    筛，不必先回 MySQL 查一遍。

    **编码结果与子块数不符时当场报错。** 用 zip 拼装的话，向量少了会
    静默丢掉末尾的子块——那些内容在 MySQL 里有、在向量库里没有，永远
    检索不到，而且不报错。
    """
    if not children:
        return []

    result = embedding.encode([chunk.text for chunk in children])
    if len(result.dense) != len(children) or len(result.sparse) != len(children):
        raise ValueError(
            "编码结果数量与子块数不符：子块 %d 条，稠密 %d 条，稀疏 %d 条"
            % (len(children), len(result.dense), len(result.sparse))
        )

    return [
        ChunkVector(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            # 子块的 parent_id 由切分保证非空，父块不会走到这里
            parent_id=chunk.parent_id or "",
            chunk_index=chunk.chunk_index,
            country=record.country or "",
            doc_type=record.doc_type or "",
            publisher=record.publisher or "",
            dense=dense,
            sparse=sparse,
        )
        for chunk, dense, sparse in zip(children, result.dense, result.sparse)
    ]


@dataclass(frozen=True)
class IndexingContext:
    """索引流程依赖的外部能力。

    集中成一个对象而不是一路传四个参数：批量导入时同一组依赖要反复传给
    每一份文档，散着传容易漏掉某个，或者把两个参数传反。
    """

    mysql: MysqlStore
    milvus: MilvusStore
    embedding: EmbeddingStore
    counter: TokenCounter


@dataclass(frozen=True)
class AnalysisResult:
    """一次解析与索引的结果。

    不带 query 动作与文件名：那些由调用方组装进 `DocumentOutcome`，
    这一步只关心「做完没有、结果如何」。
    """

    ok: bool
    detail: str
    parent_count: int = 0
    child_count: int = 0


def _attempt_summary(outcome: ParseOutcome) -> str:
    """把各层解析的失败原因压成一句话。

    人工处理时最想知道的就是「机器试过什么、卡在哪」，只写「解析失败」
    等于没写，人得从头再来一遍。
    """
    parts = [
        "%s：%s" % (attempt.tier, attempt.detail)
        for attempt in outcome.attempts
        if not attempt.ok
    ]
    return "各层解析都不合格（%s），交人工" % "；".join(parts)


def _retire_siblings(
    record: DocumentRecord, context: IndexingContext
) -> list[str]:
    """把同一链接下的其他活跃记录下线，并删掉它们的向量。

    **仅改状态拦不住检索。** Milvus 里没有 status 字段，检索按
    country / doc_type / publisher 过滤，筛不到状态，旧记录的向量照样
    会被召回——过时内容仍会出现在答案的引用里，而引用看着有出处。
    """
    if record.source_url is None:
        return []

    taken_down = context.mysql.supersede_siblings(
        record.source_url, record.doc_id
    )
    for stale_id in taken_down:
        context.milvus.delete_by_doc(stale_id)
    return taken_down


def analyze_document(
    document: CollectedDocument,
    record: DocumentRecord,
    context: IndexingContext,
    *,
    now: datetime | None = None,
) -> AnalysisResult:
    """解析、切块、向量化，把结果写进两处存储。

    进到这里说明文档已经登记过（见 `register_document`），`record` 就是
    登记时写进去的那条，`doc_id` 从它身上取。
    """
    if record.missing_fields:
        # 门禁。切片进了向量库就会被检索到并挂进引用里，而缺 source_url
        # 时拿不出可核对的出处，「可追溯」这个核心主张就断了。
        return AnalysisResult(
            ok=True,
            detail="元数据不齐（缺 %s），只登记未入库"
            % "、".join(record.missing_fields),
        )

    parsed = parse_document(
        ParseRequest(
            doc_id=record.doc_id,
            title=record.title,
            source_url=record.source_url,
            raw_path=document.raw_path,
        )
    )
    if parsed.document is None:
        context.mysql.update_document_status(
            record.doc_id, DocumentStatus.NEEDS_MANUAL, now=now
        )
        return AnalysisResult(ok=False, detail=_attempt_summary(parsed))

    chunks = chunk_document(parsed.document, context.counter)
    children = [chunk for chunk in chunks if chunk.level is ChunkLevel.CHILD]

    try:
        vectors = build_vectors(children, record, context.embedding)
        # 先清掉这份文档上一轮的向量：分块数量可能变少，光靠 upsert
        # 清不掉多出来的那几条，它们会继续被检索到。
        context.milvus.delete_by_doc(record.doc_id)
        context.milvus.upsert_chunks(vectors)
        context.mysql.replace_chunks(record.doc_id, chunks)
    except Exception:
        # 先记 failed 再抛出：调用方要知道这次没成，而库里留下痕迹才能
        # 看出这份文档试过但没跑完，否则它一直是 pending，看着像没开始
        context.mysql.update_document_status(
            record.doc_id, DocumentStatus.FAILED, now=now
        )
        raise

    context.mysql.update_document_status(
        record.doc_id, DocumentStatus.INDEXED, now=now
    )
    _retire_siblings(record, context)

    parent_count = len(chunks) - len(children)
    return AnalysisResult(
        ok=True,
        detail="入库 %d 个父块、%d 个子块" % (parent_count, len(children)),
        parent_count=parent_count,
        child_count=len(children),
    )


def _update_meta(
    document: CollectedDocument,
    prepared: PreparedDocument,
    context: IndexingContext,
    *,
    now: datetime | None,
) -> DocumentOutcome:
    """内容没变，只是清单上的登记改了。

    切块与向量只依赖正文，元数据改了重跑一遍纯属白干，所以这里只改属性。

    **但 Milvus 里冗余存着三个过滤维度，要一并同步**，否则按 publisher
    筛会漏掉这份文档——它的向量还在用旧值。
    """
    matched = prepared.decision.matched
    if matched is None:
        raise ValueError("UPDATE_META 缺少命中记录，判重结果不完整")

    record = _to_record(
        document, prepared.content_hash, matched.doc_id, DocumentStatus.INDEXED
    )
    context.mysql.update_document_meta(record, now=now)

    if record.missing_fields:
        # 改完之后反而不齐了（比如把 publisher 清空）。门禁对它的要求
        # 和第一次入库时一样：元数据不齐的文档不该留在向量库里，否则
        # 会被检索到并挂进引用，而缺 source_url 时拿不出可核对的出处。
        # 向量都删了，也就不必再同步过滤字段。
        context.milvus.delete_by_doc(matched.doc_id)
        context.mysql.update_document_status(
            matched.doc_id, DocumentStatus.PENDING, now=now
        )
        return DocumentOutcome(
            file_name=document.raw_path.name,
            action=ImportAction.UPDATE_META,
            ok=True,
            detail="元数据改后不齐（缺 %s），已移出向量库"
            % "、".join(record.missing_fields),
            doc_id=matched.doc_id,
        )

    synced = context.milvus.update_scalar_fields(
        matched.doc_id,
        country=record.country or "",
        doc_type=record.doc_type or "",
        publisher=record.publisher or "",
    )
    return DocumentOutcome(
        file_name=document.raw_path.name,
        action=ImportAction.UPDATE_META,
        ok=True,
        detail="元数据已更新，同步了 %d 条向量的过滤字段" % synced,
        doc_id=matched.doc_id,
    )


def index_document(
    document: CollectedDocument,
    context: IndexingContext,
    *,
    now: datetime | None = None,
) -> DocumentOutcome:
    """处理一份文档：判重、按判出的动作执行、汇总成一条结果。

    这是批量导入逐个文档调用的入口。**单个文档出错不在这里吞掉**——
    异常照常向上抛，由调用方决定是整批中止还是记下来继续。
    """
    prepared = prepare_document(document, context.mysql)
    action = prepared.decision.action

    if action is ImportAction.SKIP:
        matched = prepared.decision.matched
        return DocumentOutcome(
            file_name=document.raw_path.name,
            action=action,
            ok=True,
            detail="内容与清单登记都未变，跳过",
            doc_id=matched.doc_id if matched else None,
        )

    if action is ImportAction.UPDATE_META:
        return _update_meta(document, prepared, context, now=now)

    record = register_document(document, prepared, context.mysql, now=now)
    result = analyze_document(document, record, context, now=now)
    return DocumentOutcome(
        file_name=document.raw_path.name,
        action=action,
        ok=result.ok,
        detail=result.detail,
        doc_id=record.doc_id,
        parent_count=result.parent_count,
        child_count=result.child_count,
    )
