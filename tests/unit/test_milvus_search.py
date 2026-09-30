"""Milvus 混合检索与拒答判据的单元测试。

不连真实服务，假对象见 milvus_fakes.py。

这里关心的是「传下去的参数对不对」：两路的字段、度量、候选数、过滤条件、
权重，任何一个传错都不会报错，只会让检索结果静默变差。
"""

from __future__ import annotations

from typing import Any

import pytest
from pymilvus import WeightedRanker

from cbe_rag.storage.milvus_store import MilvusStore
from milvus_fakes import build_db_store, make_hit, request_details

DENSE = [0.1, 0.2, 0.3]
SPARSE = {7: 0.5, 42: 0.25}


def run_search(store: MilvusStore, **overrides: Any) -> list[Any]:
    """按默认参数跑一次混合检索，参数可按需覆盖。"""
    kwargs: dict[str, Any] = {
        "dense_limit": 20,
        "sparse_limit": 40,
        "weights": (0.7, 0.3),
        "limit": 10,
    }
    kwargs.update(overrides)
    return store.hybrid_search(DENSE, SPARSE, **kwargs)


def last_requests(db_client: Any) -> list[dict[str, Any]]:
    """取最近一次混合检索的两路请求参数。"""
    return request_details(db_client.hybrid_searched[-1]["reqs"])


class TestHybridSearch:
    def test_returns_fused_hits(self) -> None:
        store, _, _ = build_db_store(hybrid_rows=(make_hit(distance=0.7),))

        hits = run_search(store)

        assert len(hits) == 1
        assert hits[0].chunk_id == "doc-1_c0000"
        assert hits[0].doc_id == "doc-1"

    def test_keeps_the_fused_score(self) -> None:
        store, _, _ = build_db_store(hybrid_rows=(make_hit(distance=0.697),))

        assert run_search(store)[0].score == pytest.approx(0.697)

    def test_no_hits_returns_empty_list(self) -> None:
        # 没有命中不是错误
        store, _, _ = build_db_store(hybrid_rows=())

        assert run_search(store) == []

    def test_zero_limit_sends_nothing(self) -> None:
        store, db_client, _ = build_db_store()

        assert run_search(store, limit=0) == []
        assert db_client.hybrid_searched == []

    def test_queries_both_routes(self) -> None:
        store, db_client, _ = build_db_store()

        run_search(store)

        assert [detail["anns_field"] for detail in last_requests(db_client)] == [
            "dense_vector",
            "sparse_vector",
        ]

    def test_each_route_keeps_its_own_limit(self) -> None:
        # 两路候选数不同是有理由的：稀疏路分数分布更陡，多取一些
        # 给融合一次机会
        store, db_client, _ = build_db_store()

        run_search(store)

        assert [detail["limit"] for detail in last_requests(db_client)] == [20, 40]

    def test_each_route_uses_its_own_metric(self) -> None:
        # 稠密用余弦、稀疏用内积。换错了 Milvus 不报错，只是结果不对。
        store, db_client, _ = build_db_store()

        run_search(store)

        metrics = [
            detail["param"]["metric_type"]  # type: ignore[index]
            for detail in last_requests(db_client)
        ]
        assert metrics == ["COSINE", "IP"]

    def test_weights_go_to_the_ranker(self) -> None:
        store, db_client, _ = build_db_store()

        run_search(store, weights=(0.3, 0.7))

        assert isinstance(db_client.hybrid_searched[-1]["ranker"], WeightedRanker)

    def test_filter_applies_to_both_routes(self) -> None:
        # 只给一路加过滤，另一路会捞出本该被排除的文档
        store, db_client, _ = build_db_store()

        run_search(store, filter_expression='country == "EU"')

        assert [detail["expr"] for detail in last_requests(db_client)] == [
            'country == "EU"',
            'country == "EU"',
        ]

    def test_no_filter_leaves_the_expression_empty(self) -> None:
        store, db_client, _ = build_db_store()

        run_search(store)

        assert [detail["expr"] for detail in last_requests(db_client)] == [None, None]

    def test_ensures_the_collection_is_loaded(self) -> None:
        store, db_client, _ = build_db_store()

        run_search(store)

        assert db_client.loaded

    def test_requests_the_fields_folding_needs(self) -> None:
        # parent_id 漏了折叠会直接失败，不算隐蔽；doc_id 漏了要等取
        # 引用元数据时才发现，那时已经离开了这一层的排查范围
        store, db_client, _ = build_db_store()

        run_search(store)

        assert set(db_client.hybrid_searched[-1]["output_fields"]) == {
            "chunk_id",
            "doc_id",
            "parent_id",
        }


class TestTopDenseScore:
    def test_returns_the_highest_cosine(self) -> None:
        store, _, _ = build_db_store(search_rows=(make_hit(distance=0.638),))

        assert store.top_dense_score(DENSE) == pytest.approx(0.638)

    def test_no_hit_returns_none(self) -> None:
        # 没有命中时返回 None 而不是 0：0 会被读成「检索过了但都不相关」，
        # 与「一条都没检索到」是两回事
        store, _, _ = build_db_store(search_rows=())

        assert store.top_dense_score(DENSE) is None

    def test_queries_the_dense_route_only(self) -> None:
        # 拒答判据是稠密路的余弦，掺进稀疏路就没有绝对含义了
        store, db_client, _ = build_db_store()

        store.top_dense_score(DENSE)

        assert db_client.searched[-1]["anns_field"] == "dense_vector"
        assert db_client.searched[-1]["search_params"] == {"metric_type": "COSINE"}

    def test_asks_for_a_single_hit(self) -> None:
        # 只要最高分，取更多是白算
        store, db_client, _ = build_db_store()

        store.top_dense_score(DENSE)

        assert db_client.searched[-1]["limit"] == 1

    def test_passes_the_filter_through(self) -> None:
        # 过滤后的最高分才和检索用的是同一批文档
        store, db_client, _ = build_db_store()

        store.top_dense_score(DENSE, filter_expression='country == "EU"')

        assert db_client.searched[-1]["filter"] == 'country == "EU"'

    def test_ensures_the_collection_is_loaded(self) -> None:
        store, db_client, _ = build_db_store()

        store.top_dense_score(DENSE)

        assert db_client.loaded


class TestTopSparseScore:
    """稀疏路的最高分。

    它能不能当拒答判据还没定（见 docs/adr/0002-corpus-language.md），
    但取分这件事本身要和稠密路一样可控。
    """

    def test_returns_the_highest_score(self) -> None:
        store, _, _ = build_db_store(search_rows=(make_hit(distance=0.0786),))

        assert store.top_sparse_score(SPARSE) == pytest.approx(0.0786)

    def test_no_hit_returns_none(self) -> None:
        store, _, _ = build_db_store(search_rows=())

        assert store.top_sparse_score(SPARSE) is None

    def test_queries_the_sparse_route_only(self) -> None:
        # 稀疏是内积、稠密是余弦，换错了 Milvus 不报错，只是结果不对
        store, db_client, _ = build_db_store()

        store.top_sparse_score(SPARSE)

        assert db_client.searched[-1]["anns_field"] == "sparse_vector"
        assert db_client.searched[-1]["search_params"] == {"metric_type": "IP"}

    def test_asks_for_a_single_hit(self) -> None:
        store, db_client, _ = build_db_store()

        store.top_sparse_score(SPARSE)

        assert db_client.searched[-1]["limit"] == 1

    def test_passes_the_filter_through(self) -> None:
        store, db_client, _ = build_db_store()

        store.top_sparse_score(SPARSE, filter_expression='country == "EU"')

        assert db_client.searched[-1]["filter"] == 'country == "EU"'

    def test_ensures_the_collection_is_loaded(self) -> None:
        store, db_client, _ = build_db_store()

        store.top_sparse_score(SPARSE)

        assert db_client.loaded
