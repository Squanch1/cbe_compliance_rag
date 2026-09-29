"""混合检索与折叠的单元测试。

Milvus 与嵌入都用假实现，不连服务。假对象见 milvus_fakes.py 与
indexing_fakes.py。

这里有两处「传错了不报错、只是结果变差」的地方，各配了测试：过滤条件
的取值拼进表达式、两路各自的度量。另外折叠的顺序必须确定，否则同一个
问题两次跑出的引用顺序不同。
"""

from __future__ import annotations

import pytest

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.models import RetrievalQuery
from cbe_rag.retrieval.search import build_filter, fold_by_parent, search
from cbe_rag.storage.records import VectorHit
from indexing_fakes import FakeEmbeddingStore
from milvus_fakes import build_db_store, make_hit


def hit(parent_id: str, score: float, chunk_id: str = "d_c0") -> VectorHit:
    """造一条子块命中。"""
    return VectorHit(
        chunk_id=chunk_id, doc_id="d", parent_id=parent_id, score=score
    )


class TestBuildFilter:
    def test_no_conditions_yields_empty_expression(self) -> None:
        # 空表达式等于不过滤，正是「调用方没传」该有的行为
        assert build_filter(RetrievalQuery(text="问题")) == ""

    def test_single_condition(self) -> None:
        expression = build_filter(RetrievalQuery(text="q", country="EU"))

        assert expression == 'country == "EU"'

    def test_conditions_are_joined_with_and(self) -> None:
        expression = build_filter(
            RetrievalQuery(text="q", country="EU", doc_type="guideline")
        )

        assert expression == 'country == "EU" and doc_type == "guideline"'

    def test_all_three_dimensions(self) -> None:
        expression = build_filter(
            RetrievalQuery(
                text="q",
                country="DE",
                doc_type="policy",
                publisher="eu_commission",
            )
        )

        assert expression == (
            'country == "DE" and doc_type == "policy" '
            'and publisher == "eu_commission"'
        )

    def test_allows_the_shapes_the_dimensions_actually_use(self) -> None:
        # 下划线是发布机构代码里真有的（eu_commission）
        expression = build_filter(
            RetrievalQuery(text="q", publisher="eu_commission")
        )

        assert expression == 'publisher == "eu_commission"'

    def test_rejects_values_that_could_break_the_expression(self) -> None:
        # filter 没有参数占位符，值只能拼进字符串，而它来自接口层。
        # 这个输入会把过滤条件整个改掉，等于查了别的国家的文档。
        with pytest.raises(ValueError, match="非法字符"):
            build_filter(RetrievalQuery(text="q", country='EU" or country != "DE'))


class TestFoldByParent:
    def test_same_parent_keeps_the_highest_score(self) -> None:
        # 求平均会被不相关的子块拉低，反而惩罚了覆盖得广的父块
        hits = [hit("p1", 0.4, "d_c0"), hit("p1", 0.7, "d_c1")]

        folded = fold_by_parent(hits)

        assert len(folded) == 1
        assert folded[0].score == pytest.approx(0.7)

    def test_counts_matched_children(self) -> None:
        hits = [hit("p1", 0.4, "d_c0"), hit("p1", 0.7, "d_c1")]

        assert fold_by_parent(hits)[0].matched_children == 2

    def test_keeps_different_parents_separate(self) -> None:
        hits = [hit("p1", 0.4), hit("p2", 0.7)]

        assert len(fold_by_parent(hits)) == 2

    def test_sorted_by_score_descending(self) -> None:
        hits = [hit("p1", 0.4), hit("p2", 0.9), hit("p3", 0.6)]

        assert [item.parent_id for item in fold_by_parent(hits)] == [
            "p2",
            "p3",
            "p1",
        ]

    def test_ties_are_broken_deterministically(self) -> None:
        # 顺序随插入顺序变的话，同一个问题两次跑出的引用顺序不同，
        # 出了问题没法复现
        hits = [hit("p2", 0.5), hit("p1", 0.5)]

        assert [item.parent_id for item in fold_by_parent(hits)] == ["p1", "p2"]

    def test_empty_input_yields_empty_list(self) -> None:
        assert fold_by_parent([]) == []


class TestSearch:
    def test_encodes_the_query_text(self) -> None:
        store, _, _ = build_db_store()
        embedding = FakeEmbeddingStore()

        search(
            RetrievalQuery(text="进口一站式服务的上限"),
            milvus=store,
            embedding=embedding,
            config=RetrievalConfig(),
        )

        assert embedding.encoded == [["进口一站式服务的上限"]]

    def test_passes_the_configured_route_limits(self) -> None:
        store, db_client, _ = build_db_store()

        search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        from milvus_fakes import request_details

        details = request_details(db_client.hybrid_searched[-1]["reqs"])
        assert [detail["limit"] for detail in details] == [20, 40]

    def test_fetches_all_candidates_before_folding(self) -> None:
        # 折叠会让条数变少，先砍一批再折叠，剩下的父块可能不够
        store, db_client, _ = build_db_store()

        search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        assert db_client.hybrid_searched[-1]["limit"] == 60

    def test_folds_hits_into_parents(self) -> None:
        store, _, _ = build_db_store(
            hybrid_rows=(
                make_hit(chunk_id="d_c0", parent_id="p1", distance=0.4),
                make_hit(chunk_id="d_c1", parent_id="p1", distance=0.7),
                make_hit(chunk_id="d_c2", parent_id="p2", distance=0.5),
            )
        )

        outcome = search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        assert [item.parent_id for item in outcome.parents] == ["p1", "p2"]
        assert outcome.parents[0].score == pytest.approx(0.7)

    def test_truncates_to_the_configured_parent_count(self) -> None:
        store, _, _ = build_db_store(
            hybrid_rows=tuple(
                make_hit(chunk_id="d_c%d" % i, parent_id="p%d" % i, distance=0.9 - i * 0.1)
                for i in range(8)
            )
        )

        outcome = search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(context_parents=3),
        )

        assert len(outcome.parents) == 3

    def test_returns_the_dense_top_score(self) -> None:
        store, _, _ = build_db_store(search_rows=(make_hit(distance=0.638),))

        outcome = search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        assert outcome.dense_top_score == pytest.approx(0.638)

    def test_filter_reaches_both_routes_and_the_score_probe(self) -> None:
        # 三处用同一个过滤条件，漏一处就会拿到不该出现的文档
        store, db_client, _ = build_db_store()

        search(
            RetrievalQuery(text="q", country="EU"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        from milvus_fakes import request_details

        details = request_details(db_client.hybrid_searched[-1]["reqs"])
        assert [detail["expr"] for detail in details] == [
            'country == "EU"',
            'country == "EU"',
        ]
        assert db_client.searched[-1]["filter"] == 'country == "EU"'

    def test_no_hits_yields_no_parents(self) -> None:
        store, _, _ = build_db_store(hybrid_rows=(), search_rows=())

        outcome = search(
            RetrievalQuery(text="q"),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            config=RetrievalConfig(),
        )

        assert outcome.parents == []
        assert outcome.dense_top_score is None
