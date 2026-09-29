"""按 parent_id 恢复父块的单元测试。

MySQL 用假实现，不连服务。

两条数据不一致的路径各有一条测试：Milvus 里有子块指向某个父块、而 MySQL
里没有；以及父块在、文档记录不在。两种都是索引中途断掉的痕迹，跳过不报
会让答案缺一块而不留痕。
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from cbe_rag.ingestion.parser.schema import Chunk, ChunkLevel
from cbe_rag.retrieval.context import ContextError, load_parents
from cbe_rag.retrieval.models import ParentHit
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

TODAY = date(2026, 9, 29)
LONG_TEXT = "进口一站式服务适用于价值不超过 150 欧元的货物，卖家在销售时代收增值税。"


def make_parent_chunk(**overrides: Any) -> Chunk:
    """造一个父块。"""
    fields: dict[str, Any] = {
        "chunk_id": "doc-1_p0000",
        "doc_id": "doc-1",
        "level": ChunkLevel.PARENT,
        "chunk_index": 0,
        "text": LONG_TEXT,
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


def make_hit(**overrides: Any) -> ParentHit:
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

    记录每次查询的 id 列表，用来验证批量而非逐条。
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


class TestLoadParents:
    def test_restores_the_parent_text(self) -> None:
        # Milvus 里只有子块的向量，父块正文得用 parent_id 回 MySQL 取
        parents = load_parents([make_hit()], make_mysql())

        assert len(parents) == 1
        assert parents[0].text == LONG_TEXT

    def test_carries_the_citation_metadata(self) -> None:
        parents = load_parents([make_hit()], make_mysql())

        assert parents[0].title == "欧洲增值税常见问题"
        assert parents[0].source_url is not None
        assert parents[0].country == "EU"
        assert parents[0].doc_type == "faq"

    def test_carries_the_score(self) -> None:
        # 分数是折叠时算的，恢复父块不该把它弄丢
        parents = load_parents([make_hit(score=0.83)], make_mysql())

        assert parents[0].score == pytest.approx(0.83)

    def test_keeps_the_incoming_order(self) -> None:
        # 传进来的顺序就是相关性顺序，返回时不能被打乱
        hits = [
            make_hit(parent_id="doc-1_p0001", score=0.9),
            make_hit(parent_id="doc-1_p0000", score=0.5),
        ]
        mysql = make_mysql(
            chunks=[
                make_parent_chunk(chunk_id="doc-1_p0000"),
                make_parent_chunk(chunk_id="doc-1_p0001"),
            ]
        )

        parents = load_parents(hits, mysql)

        assert [parent.parent_id for parent in parents] == [
            "doc-1_p0001",
            "doc-1_p0000",
        ]

    def test_empty_hits_queries_nothing(self) -> None:
        mysql = make_mysql()

        assert load_parents([], mysql) == []
        assert mysql.chunk_lookups == []
        assert mysql.document_lookups == []

    def test_queries_in_two_batches(self) -> None:
        # 5 个父块逐条查就是 10 次往返，而它们本来就是两批
        hits = [
            make_hit(parent_id="doc-1_p0000"),
            make_hit(parent_id="doc-1_p0001"),
        ]
        mysql = make_mysql(
            chunks=[
                make_parent_chunk(chunk_id="doc-1_p0000"),
                make_parent_chunk(chunk_id="doc-1_p0001"),
            ]
        )

        load_parents(hits, mysql)

        assert len(mysql.chunk_lookups) == 1
        assert len(mysql.document_lookups) == 1

    def test_looks_up_each_document_once(self) -> None:
        # 同一份文档的多个父块被命中是常事，doc_id 要去重
        hits = [
            make_hit(parent_id="doc-1_p0000"),
            make_hit(parent_id="doc-1_p0001"),
        ]
        mysql = make_mysql(
            chunks=[
                make_parent_chunk(chunk_id="doc-1_p0000"),
                make_parent_chunk(chunk_id="doc-1_p0001"),
            ]
        )

        load_parents(hits, mysql)

        assert mysql.document_lookups == [["doc-1"]]

    def test_different_documents_are_all_fetched(self) -> None:
        hits = [
            make_hit(parent_id="doc-1_p0000", doc_id="doc-1"),
            make_hit(parent_id="doc-2_p0000", doc_id="doc-2"),
        ]
        mysql = make_mysql(
            chunks=[
                make_parent_chunk(chunk_id="doc-1_p0000", doc_id="doc-1"),
                make_parent_chunk(chunk_id="doc-2_p0000", doc_id="doc-2"),
            ],
            documents=[make_document(doc_id="doc-1"), make_document(doc_id="doc-2")],
        )

        parents = load_parents(hits, mysql)

        assert len(parents) == 2


class TestInconsistentData:
    def test_missing_parent_chunk_raises(self) -> None:
        # Milvus 里有子块指向它、MySQL 里没有：索引中途断了
        with pytest.raises(ContextError, match="父块"):
            load_parents([make_hit()], make_mysql(chunks=[]))

    def test_missing_parent_error_names_the_id(self) -> None:
        # 只说「缺了一个父块」，排查时还得自己去比对
        with pytest.raises(ContextError, match="doc-1_p0000"):
            load_parents([make_hit()], make_mysql(chunks=[]))

    def test_missing_document_raises(self) -> None:
        # 引用要带出处与生效日期，缺了就没法生成合规的引用
        with pytest.raises(ContextError, match="元数据"):
            load_parents([make_hit()], make_mysql(documents=[]))

    def test_missing_document_error_names_the_id(self) -> None:
        with pytest.raises(ContextError, match="doc-1"):
            load_parents([make_hit()], make_mysql(documents=[]))
