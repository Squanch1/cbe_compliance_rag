"""检索层测试共用的假对象与构造辅助。

单独放一个模块而不是留在某个测试文件里：父块恢复、编排两组测试都要用。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel
from cbe_rag.retrieval.models import ParentHit
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

TODAY = date(2026, 9, 29)
PARENT_TEXT = (
    "进口一站式服务适用于价值不超过 150 欧元的货物，卖家在销售时代收增值税。"
)


def make_parent_chunk(**overrides: Any) -> Chunk:
    """造一个父块。"""
    fields: dict[str, Any] = {
        "chunk_id": "doc-1_p0000",
        "doc_id": "doc-1",
        "level": ChunkLevel.PARENT,
        "chunk_index": 0,
        "text": PARENT_TEXT,
        "token_count": 40,
    }
    fields.update(overrides)
    return Chunk(**fields)


def make_document(**overrides: Any) -> DocumentRecord:
    """造一条文档记录。"""
    fields: dict[str, Any] = {
        "doc_id": "doc-1",
        "content_hash": "a" * 64,
        "status": DocumentStatus.INDEXED,
        "title": "欧洲增值税常见问题",
        "platform": "amazon",
        "source_url": "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX",
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "collected_date": TODAY,
        "raw_path": "C:/data/raw/a.html",
    }
    fields.update(overrides)
    return DocumentRecord(**fields)


def make_parent_hit(**overrides: Any) -> ParentHit:
    """造一条折叠后的父块命中。"""
    fields: dict[str, Any] = {
        "parent_id": "doc-1_p0000",
        "doc_id": "doc-1",
        "score": 0.7,
        "matched_children": 1,
    }
    fields.update(overrides)
    return ParentHit(**fields)


class FakeMysql:
    """只实现取分块与取文档两个查询的假适配器。

    记录每次查询的 id 列表，用来验证是批量而不是逐条。
    """

    def __init__(
        self,
        chunks: list[Chunk] | None = None,
        documents: list[DocumentRecord] | None = None,
    ) -> None:
        self._chunks = chunks if chunks is not None else []
        self._documents = documents if documents is not None else []
        self.chunk_lookups: list[list[str]] = []
        self.document_lookups: list[list[str]] = []

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        self.chunk_lookups.append(list(chunk_ids))
        return [chunk for chunk in self._chunks if chunk.chunk_id in chunk_ids]

    def get_documents(self, doc_ids: list[str]) -> list[DocumentRecord]:
        self.document_lookups.append(list(doc_ids))
        return [record for record in self._documents if record.doc_id in doc_ids]


def make_mysql(**overrides: Any) -> FakeMysql:
    """造一份「父块与文档都在」的假库，可只替换其中一项。"""
    fields: dict[str, Any] = {
        "chunks": [make_parent_chunk()],
        "documents": [make_document()],
    }
    fields.update(overrides)
    return FakeMysql(**fields)
