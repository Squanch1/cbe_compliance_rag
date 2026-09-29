"""按 parent_id 恢复父块的单元测试。

MySQL 用假实现，不连服务。

两条数据不一致的路径各有一条测试：Milvus 里有子块指向某个父块、而 MySQL
里没有；以及父块在、文档记录不在。两种都是索引中途断掉的痕迹，跳过不报
会让答案缺一块而不留痕。
"""

from __future__ import annotations

import pytest

from cbe_rag.retrieval.context import ContextError, load_parents
from retrieval_fakes import (
    PARENT_TEXT,
    make_document,
    make_mysql,
    make_parent_chunk,
    make_parent_hit,
)


class TestLoadParents:
    def test_restores_the_parent_text(self) -> None:
        # Milvus 里只有子块的向量，父块正文得用 parent_id 回 MySQL 取
        parents = load_parents([make_parent_hit()], make_mysql())

        assert len(parents) == 1
        assert parents[0].text == PARENT_TEXT

    def test_carries_the_citation_metadata(self) -> None:
        parents = load_parents([make_parent_hit()], make_mysql())

        assert parents[0].title == "欧洲增值税常见问题"
        assert parents[0].source_url is not None
        assert parents[0].country == "EU"
        assert parents[0].doc_type == "faq"

    def test_carries_the_score(self) -> None:
        # 分数是折叠时算的，恢复父块不该把它弄丢
        parents = load_parents([make_parent_hit(score=0.83)], make_mysql())

        assert parents[0].score == pytest.approx(0.83)

    def test_keeps_the_incoming_order(self) -> None:
        # 传进来的顺序就是相关性顺序，返回时不能被打乱
        hits = [
            make_parent_hit(parent_id="doc-1_p0001", score=0.9),
            make_parent_hit(parent_id="doc-1_p0000", score=0.5),
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
            make_parent_hit(parent_id="doc-1_p0000"),
            make_parent_hit(parent_id="doc-1_p0001"),
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
            make_parent_hit(parent_id="doc-1_p0000"),
            make_parent_hit(parent_id="doc-1_p0001"),
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
            make_parent_hit(parent_id="doc-1_p0000", doc_id="doc-1"),
            make_parent_hit(parent_id="doc-2_p0000", doc_id="doc-2"),
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
            load_parents([make_parent_hit()], make_mysql(chunks=[]))

    def test_missing_parent_error_names_the_id(self) -> None:
        # 只说「缺了一个父块」，排查时还得自己去比对
        with pytest.raises(ContextError, match="doc-1_p0000"):
            load_parents([make_parent_hit()], make_mysql(chunks=[]))

    def test_missing_document_raises(self) -> None:
        # 引用要带出处与生效日期，缺了就没法生成合规的引用
        with pytest.raises(ContextError, match="元数据"):
            load_parents([make_parent_hit()], make_mysql(documents=[]))

    def test_missing_document_error_names_the_id(self) -> None:
        with pytest.raises(ContextError, match="doc-1"):
            load_parents([make_parent_hit()], make_mysql(documents=[]))
