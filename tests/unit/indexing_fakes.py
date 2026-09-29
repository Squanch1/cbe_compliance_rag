"""索引流程测试共用的假对象与构造辅助。

单独放一个模块而不是留在某个测试文件里：判重、编排、批量导入几组测试
都要用，从某个测试文件 import 会让「哪个是测试、哪个是辅助」变含糊。
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

from cbe_rag.ingestion.fetcher.collector import CollectedDocument
from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel, DocumentMeta
from cbe_rag.indexing.pipeline import (
    AnalysisResult,
    IndexingContext,
    analyze_document,
    prepare_document,
    register_document,
)
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.embedding import EmbeddingResult
from cbe_rag.storage.records import ChunkVector, DocumentRecord

TODAY = date(2026, 9, 29)
SOURCE_URL = "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX"

# 正文要够长：质量门禁有一条「总字符数不低于 200」的下限，用于拦住
# 扫描件和解析出空壳的情况。样例太短会被当成解析失败。
HTML_WITH_CONTENT = """<!doctype html>
<html><body><div id="help-content">
<h1>Import One-Stop Shop</h1>
<p>进口一站式服务（Import One-Stop Shop，IOSS）自 2021 年 7 月 1 日起实施，
适用于从第三国或第三地区向欧盟境内消费者销售、且价值不超过 150 欧元的货物。
卖家在销售时代收增值税并通过 IOSS 申报，无需在每一个销售目的国分别注册税号。</p>
<p>对于价值超过 150 欧元的货物不适用 IOSS，卖家需要按常规进口流程办理，
由买方在进口时缴纳进口增值税。此时卖家应当在货物存放地或销售目的国
注册增值税号，并按当地规定申报。</p>
<p>注册 IOSS 需要提供企业注册信息、税务识别号，以及在欧盟境内的中介信息。
使用亚马逊物流的卖家可以直接使用亚马逊提供的 IOSS 注册号，
具体位置在卖家平台的税务设置页面。</p>
</div></body></html>
"""

MANIFEST_COLUMNS = (
    "file_name",
    "title",
    "source_url",
    "publisher",
    "country",
    "doc_type",
    "effective_date",
    "platform",
    "notes",
)


def write_document_file(
    tmp_path: Path,
    content: str = "欧洲增值税常见问题",
    name: str = "amazon-eu-vat-faq.html",
) -> Path:
    """写一个真实的文件，哈希读的就是它。"""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def write_html_file(tmp_path: Path, body: str = HTML_WITH_CONTENT) -> Path:
    """写一个能被解析的 HTML 文件。

    正文放在 help-content 容器里：站点选择器按域名配置，
    sellercentral.amazon.com 只认这个容器，放在别处会被判成空文档。
    """
    path = tmp_path / "amazon-eu-vat-faq.html"
    path.write_text(body, encoding="utf-8")
    return path


def make_document(raw_path: Path, **overrides: Any) -> CollectedDocument:
    """造一份采集到的文档，元数据字段可按需覆盖。"""
    fields: dict[str, Any] = {
        "doc_id": "new-doc-id",
        "title": "欧洲增值税常见问题",
        "collected_date": TODAY,
        "source_url": SOURCE_URL,
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "platform": "amazon",
    }
    fields.update(overrides)
    return CollectedDocument(raw_path=raw_path, meta=DocumentMeta(**fields))


def make_record(content_hash: str, **overrides: Any) -> DocumentRecord:
    """造一条库里的记录，字段可按需覆盖。"""
    fields: dict[str, Any] = {
        "doc_id": "existing-doc-id",
        "content_hash": content_hash,
        "status": DocumentStatus.INDEXED,
        "title": "欧洲增值税常见问题",
        "platform": "amazon",
        "source_url": SOURCE_URL,
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "collected_date": TODAY,
        "raw_path": "C:/data/raw/amazon-eu-vat-faq.html",
    }
    fields.update(overrides)
    return DocumentRecord(**fields)


def make_child(**overrides: Any) -> Chunk:
    """造一个子块。"""
    fields: dict[str, Any] = {
        "chunk_id": "doc-1_c0000",
        "doc_id": "doc-1",
        "parent_id": "doc-1_p0000",
        "level": ChunkLevel.CHILD,
        "chunk_index": 0,
        "text": "进口一站式服务适用于价值不超过 150 欧元的货物。",
        "token_count": 20,
        "start_offset": 0,
        "end_offset": 25,
    }
    fields.update(overrides)
    return Chunk(**fields)


def manifest_row(file_name: str, **overrides: str) -> dict[str, str]:
    """造清单里的一行，列名与 manifest.py 的定义对齐。"""
    row = {
        "file_name": file_name,
        "title": "欧洲增值税常见问题",
        "source_url": SOURCE_URL,
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": "",
        "platform": "amazon",
        "notes": "",
    }
    row.update(overrides)
    return row


def write_manifest(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    """写一份清单文件。未给的列留空。"""
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in MANIFEST_COLUMNS})
    return path


class FakeMysqlStore:
    """无状态的假 MySQL 适配器：库里什么都没有，也就是全新文档。

    记录查询顺序，用来验证「哈希命中时不该再查 URL」这个取舍。
    """

    def __init__(
        self,
        by_hash: DocumentRecord | None = None,
        by_url: DocumentRecord | None = None,
    ) -> None:
        self._by_hash = by_hash
        self._by_url = by_url
        self.queries: list[str] = []

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        self.queries.append("hash")
        return self._by_hash

    def find_active_by_source_url(self, source_url: str) -> DocumentRecord | None:
        self.queries.append("url")
        return self._by_url


class RecordingMysqlStore:
    """记录读写操作的假适配器，不连服务。

    查询结果按需注入：不传就是「库里什么都没有」，也就是全新文档。
    """

    def __init__(
        self,
        siblings: list[str] | None = None,
        by_hash: DocumentRecord | None = None,
        by_url: DocumentRecord | None = None,
        active: list[DocumentRecord] | None = None,
    ) -> None:
        self.inserted: list[DocumentRecord] = []
        self.meta_updates: list[DocumentRecord] = []
        self.status_updates: list[tuple[str, DocumentStatus]] = []
        self.chunks_written: list[tuple[str, list[Chunk]]] = []
        self.parse_attempts_written: list[tuple[str, list[dict[str, Any]]]] = []
        self.sibling_lookups: list[tuple[str, str]] = []
        self._siblings = siblings if siblings is not None else []
        self._by_hash = by_hash
        self._by_url = by_url
        self._active = active if active is not None else []

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        return self._by_hash

    def find_active_by_source_url(self, source_url: str) -> DocumentRecord | None:
        return self._by_url

    def list_active_documents(self) -> list[DocumentRecord]:
        return list(self._active)

    def insert_document(
        self, record: DocumentRecord, *, now: Any = None
    ) -> None:
        self.inserted.append(record)

    def update_document_meta(
        self, record: DocumentRecord, *, now: Any = None
    ) -> None:
        self.meta_updates.append(record)

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        *,
        parse_attempts: list[dict[str, Any]] | None = None,
        now: Any = None,
    ) -> None:
        self.status_updates.append((doc_id, status))
        if parse_attempts is not None:
            self.parse_attempts_written.append((doc_id, parse_attempts))

    def replace_chunks(
        self, doc_id: str, chunks: list[Chunk], *, now: Any = None
    ) -> None:
        self.chunks_written.append((doc_id, list(chunks)))

    def supersede_siblings(
        self, source_url: str, keep_doc_id: str, *, now: Any = None
    ) -> list[str]:
        self.sibling_lookups.append((source_url, keep_doc_id))
        return list(self._siblings)


class RecordingMilvusStore:
    """记录写操作的假 Milvus 适配器，不连服务。"""

    def __init__(self, synced: int = 1) -> None:
        self.deleted: list[str] = []
        self.upserted: list[list[ChunkVector]] = []
        self.scalar_updates: list[dict[str, Any]] = []
        self._synced = synced

    def delete_by_doc(self, doc_id: str) -> None:
        self.deleted.append(doc_id)

    def upsert_chunks(self, vectors: list[ChunkVector]) -> int:
        self.upserted.append(list(vectors))
        return len(vectors)

    def update_scalar_fields(
        self, doc_id: str, *, country: str, doc_type: str, publisher: str
    ) -> int:
        self.scalar_updates.append(
            {
                "doc_id": doc_id,
                "country": country,
                "doc_type": doc_type,
                "publisher": publisher,
            }
        )
        return self._synced


class FakeEmbeddingStore:
    """假的嵌入适配器。

    返回的向量带上序号（第 n 条就是 [n, n, n]），这样能验证向量与文本
    没有错位——错位不会报错，只会让检索结果整体偏掉。
    """

    def __init__(self, vector_count: int | None = None) -> None:
        self.encoded: list[list[str]] = []
        self._vector_count = vector_count

    def encode(self, texts: list[str]) -> EmbeddingResult:
        self.encoded.append(list(texts))
        count = self._vector_count if self._vector_count is not None else len(texts)
        return EmbeddingResult(
            dense=[[float(index)] * 3 for index in range(count)],
            sparse=[{index: 1.0} for index in range(count)],
        )


class FakeCounter:
    """按字符数折算的假分词计数。

    切分只关心各块之间的相对大小，字符数够用，而且省掉加载真实分词器
    的那一秒。
    """

    def count(self, text: str) -> int:
        return len(text)

    def count_all(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def make_context(
    mysql: RecordingMysqlStore | None = None,
    milvus: RecordingMilvusStore | None = None,
    embedding: FakeEmbeddingStore | None = None,
) -> IndexingContext:
    """造一份索引依赖，三个假实现都可按需替换。"""
    return IndexingContext(
        mysql=mysql if mysql is not None else RecordingMysqlStore(),
        milvus=milvus if milvus is not None else RecordingMilvusStore(),
        embedding=embedding if embedding is not None else FakeEmbeddingStore(),
        counter=FakeCounter(),
    )


def register(
    tmp_path: Path,
    mysql: RecordingMysqlStore,
    *,
    by_hash: DocumentRecord | None = None,
    by_url: DocumentRecord | None = None,
    **meta_overrides: Any,
) -> DocumentRecord:
    """走一遍「准备 + 登记」，返回登记进去的记录。"""
    raw = write_document_file(tmp_path)
    document = make_document(raw, **meta_overrides)
    prepared = prepare_document(
        document, FakeMysqlStore(by_hash=by_hash, by_url=by_url)
    )
    return register_document(document, prepared, mysql)


def analyze(
    tmp_path: Path,
    context: IndexingContext,
    *,
    body: str = HTML_WITH_CONTENT,
    **meta_overrides: Any,
) -> tuple[AnalysisResult, DocumentRecord]:
    """走一遍「准备 + 登记 + 解析入库」，返回结果与登记进去的记录。"""
    document = make_document(write_html_file(tmp_path, body), **meta_overrides)
    prepared = prepare_document(document, FakeMysqlStore())
    record = register_document(document, prepared, context.mysql)
    return analyze_document(document, record, context), record
