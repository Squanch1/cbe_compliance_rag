"""问答编排的单元测试。

Milvus、MySQL、嵌入、模型全部用假实现，不连服务。

关键是两条路径的分界：质量不够时**不能调模型**。调了再让它别说，等于
先付一次钱、再把材料塞进上下文，然后指望它听话。
"""

from __future__ import annotations

from typing import Any

import pytest

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.generation.service import REFUSAL_TEXT, answer_question
from cbe_rag.retrieval.models import RetrievalQuery
from cbe_rag.retrieval.service import UncalibratedThresholdError
from cbe_rag.storage.bailian_client import Completion
from indexing_fakes import FakeEmbeddingStore
from milvus_fakes import build_db_store, make_hit
from retrieval_fakes import make_mysql

QUESTION = "进口一站式服务的适用金额上限是多少"


class FakeClient:
    """假的模型客户端，记录有没有被调用过。"""

    def __init__(self, text: str = "上限是 150 欧元 [1]。") -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(text=self._text, model="fake")


def make_config(threshold: float | None = 0.5) -> RetrievalConfig:
    """造一份检索配置。"""
    return RetrievalConfig(refuse_threshold=threshold)


def build_env(dense_top: float = 0.7) -> tuple[Any, Any]:
    """造一套「检索有命中」的假环境。"""
    return build_db_store(
        hybrid_rows=(make_hit(parent_id="doc-1_p0000", distance=0.7),),
        search_rows=(make_hit(distance=dense_top),),
    )


def _query() -> RetrievalQuery:
    """造一个检索请求。"""
    return RetrievalQuery(text=QUESTION)


class TestAnswerQuestion:
    def test_calls_the_model_when_evidence_is_sufficient(self) -> None:
        store, _, _ = build_env(dense_top=0.7)
        client = FakeClient()

        result = answer_question(
            _query(),
            config=make_config(),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=client,
        )

        assert len(client.calls) == 1
        assert result.refused is False

    def test_returns_the_generated_text(self) -> None:
        store, _, _ = build_env(dense_top=0.7)

        result = answer_question(
            _query(),
            config=make_config(),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=FakeClient("上限是 150 欧元 [1]。"),
        )

        assert "150 欧元" in result.answer.text

    def test_does_not_call_the_model_when_below_threshold(self) -> None:
        # 调了再让它别说，等于先付一次钱、再把材料塞进上下文，
        # 然后指望它听话
        store, _, _ = build_env(dense_top=0.3)
        client = FakeClient()

        result = answer_question(
            _query(),
            config=make_config(threshold=0.5),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=client,
        )

        assert client.calls == []
        assert result.refused is True

    def test_refusal_text_is_shown(self) -> None:
        store, _, _ = build_env(dense_top=0.3)

        result = answer_question(
            _query(),
            config=make_config(threshold=0.5),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=FakeClient(),
        )

        assert result.answer.text == REFUSAL_TEXT

    def test_refusal_note_carries_the_numbers(self) -> None:
        # 排查时要看「多少分、阈值多少」，只说「拒答了」没法判断该调哪边
        store, _, _ = build_env(dense_top=0.3)

        result = answer_question(
            _query(),
            config=make_config(threshold=0.5),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=FakeClient(),
        )

        assert any("0.5" in note for note in result.notes)

    def test_keeps_the_retrieval_result(self) -> None:
        # 不管答没答，召回了什么都要留着——那是排查的依据
        store, _, _ = build_env(dense_top=0.3)

        result = answer_question(
            _query(),
            config=make_config(threshold=0.5),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=FakeClient(),
        )

        assert result.retrieval.top_score == pytest.approx(0.3)

    def test_uncalibrated_threshold_raises(self) -> None:
        # 阈值没标定时不放行，这是配置里写死的约定
        store, _, _ = build_env(dense_top=0.7)

        with pytest.raises(UncalibratedThresholdError):
            answer_question(
                _query(),
                config=make_config(threshold=None),
                mysql=make_mysql(),
                milvus=store,
                embedding=FakeEmbeddingStore(),
                client=FakeClient(),
            )

    def test_no_hits_is_refused_without_calling_the_model(self) -> None:
        store, _, _ = build_db_store(hybrid_rows=(), search_rows=())
        client = FakeClient()

        result = answer_question(
            _query(),
            config=make_config(),
            mysql=make_mysql(),
            milvus=store,
            embedding=FakeEmbeddingStore(),
            client=client,
        )

        assert result.refused is True
        assert client.calls == []
