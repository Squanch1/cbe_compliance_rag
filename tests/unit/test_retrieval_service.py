"""检索编排与拒答判定的单元测试。

三个依赖都用假实现，不连服务。假对象见 milvus_fakes.py、retrieval_fakes.py
与 indexing_fakes.py。
"""

from __future__ import annotations

from typing import Any

import pytest

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.models import RetrievalQuery, RetrievalResult
from cbe_rag.retrieval.service import (
    UncalibratedThresholdError,
    passes_prefilter,
    retrieve,
)
from indexing_fakes import FakeEmbeddingStore
from milvus_fakes import build_db_store, make_hit
from retrieval_fakes import (
    PARENT_TEXT,
    make_child_chunk,
    make_mysql,
    make_parent_chunk,
)


def make_config(**overrides: Any) -> RetrievalConfig:
    """造一份检索配置，字段可按需覆盖。"""
    fields: dict[str, Any] = {"refuse_threshold": 0.45}
    fields.update(overrides)
    return RetrievalConfig(**fields)


def make_result(**overrides: Any) -> RetrievalResult:
    """造一份检索结果。"""
    fields: dict[str, Any] = {"parents": [], "top_score": 0.6}
    fields.update(overrides)
    return RetrievalResult(**fields)


class TestRetrieve:
    def test_returns_parents_restored_from_mysql(self) -> None:
        # Milvus 只给 id，正文要按 parent_id 回 MySQL 取
        store, _, _ = build_db_store(
            hybrid_rows=(make_hit(parent_id="doc-1_p0000", distance=0.7),)
        )

        result = retrieve(
            RetrievalQuery(text="q"),
            milvus=store,
            mysql=make_mysql(),
            embedding=FakeEmbeddingStore(),
            config=make_config(),
        )

        assert len(result.parents) == 1
        assert result.parents[0].text == PARENT_TEXT

    def test_keeps_the_score_order(self) -> None:
        store, _, _ = build_db_store(
            hybrid_rows=(
                make_hit(chunk_id="d_c0", parent_id="doc-1_p0001", distance=0.9),
                make_hit(chunk_id="d_c1", parent_id="doc-1_p0000", distance=0.5),
            )
        )
        mysql = make_mysql(
            chunks=[
                make_parent_chunk(chunk_id="doc-1_p0000"),
                make_parent_chunk(chunk_id="doc-1_p0001"),
                make_child_chunk(chunk_id="d_c0", parent_id="doc-1_p0001"),
                make_child_chunk(chunk_id="d_c1", parent_id="doc-1_p0000"),
            ]
        )

        result = retrieve(
            RetrievalQuery(text="q"),
            milvus=store,
            mysql=mysql,
            embedding=FakeEmbeddingStore(),
            config=make_config(),
        )

        assert [parent.parent_id for parent in result.parents] == [
            "doc-1_p0001",
            "doc-1_p0000",
        ]

    def test_carries_the_quality_metric(self) -> None:
        # 判据来自稠密路的最高余弦，不是融合分
        store, _, _ = build_db_store(search_rows=(make_hit(distance=0.638),))

        result = retrieve(
            RetrievalQuery(text="q"),
            milvus=store,
            mysql=make_mysql(),
            embedding=FakeEmbeddingStore(),
            config=make_config(),
        )

        assert result.top_score == pytest.approx(0.638)

    def test_no_hits_yields_empty_parents(self) -> None:
        store, _, _ = build_db_store(hybrid_rows=(), search_rows=())
        mysql = make_mysql()

        result = retrieve(
            RetrievalQuery(text="q"),
            milvus=store,
            mysql=mysql,
            embedding=FakeEmbeddingStore(),
            config=make_config(),
        )

        assert result.parents == []
        assert result.top_score is None
        assert mysql.chunk_lookups == []


class TestIsEvidenceSufficient:
    def test_uncalibrated_threshold_raises(self) -> None:
        # 未经校准的结果流到生成层，得到的是「看着有出处、其实没检索到
        # 相关内容」的答案，比直接拒答危险得多
        with pytest.raises(UncalibratedThresholdError, match="尚未标定"):
            passes_prefilter(make_result(), make_config(refuse_threshold=None))

    def test_score_above_threshold(self) -> None:
        assert passes_prefilter(
            make_result(top_score=0.6), make_config(refuse_threshold=0.45)
        )

    def test_score_below_threshold(self) -> None:
        assert not passes_prefilter(
            make_result(top_score=0.3), make_config(refuse_threshold=0.45)
        )

    def test_score_exactly_at_threshold(self) -> None:
        # 边界取「够」，与「不低于阈值」的表述一致
        assert passes_prefilter(
            make_result(top_score=0.45), make_config(refuse_threshold=0.45)
        )

    def test_no_hits_is_never_sufficient(self) -> None:
        # 一条都没检索到，阈值再低也不该放行
        assert not passes_prefilter(
            make_result(top_score=None), make_config(refuse_threshold=0.0)
        )
