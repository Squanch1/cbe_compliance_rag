"""问答用例的单元测试。

五个依赖全部用假实现，不连服务。假对象见 milvus_fakes.py、retrieval_fakes.py、
indexing_fakes.py 与 redis_fakes.py。

两条路径要分清：单轮走缓存，多轮不走。多轮的回答依赖前面的对话内容，
同一个问题在不同上下文里答案可能不同，用同一个键缓存会串。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from cbe_rag.config.settings import RetrievalConfig
from cbe_rag.retrieval.service import UncalibratedThresholdError
from cbe_rag.services.qa import QaContext, QaRequest, answer
from cbe_rag.storage.bailian_client import Completion
from cbe_rag.storage.milvus_store import MilvusStore
from indexing_fakes import FakeEmbeddingStore
from milvus_fakes import FakeMilvusDbClient, build_db_store, make_hit
from redis_fakes import FakeRedis, build_store
from retrieval_fakes import FakeMysql, make_document, make_mysql

QUESTION = "进口一站式服务的适用金额上限是多少"
CITABLE_ANSWER = "上限是 150 欧元 [1]。"
UNCITED_ANSWER = "上限是 150 欧元。"


class FakeBailianClient:
    """假的模型客户端，记录被调用了几次。"""

    def __init__(self, text: str = CITABLE_ANSWER) -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return Completion(text=self._text, model="fake")


@dataclass
class Env:
    """一套装好的假环境，测试直接取用其中的组件做断言。"""

    context: QaContext
    mysql: FakeMysql
    milvus: MilvusStore
    milvus_client: FakeMilvusDbClient
    embedding: FakeEmbeddingStore
    client: FakeBailianClient
    redis: FakeRedis


def make_env(
    *, dense_top: float = 0.7, model_text: str = CITABLE_ANSWER, threshold: float = 0.5
) -> Env:
    """造一套「检索有命中、材料齐全」的假环境。"""
    milvus, milvus_client, _ = build_db_store(
        hybrid_rows=(make_hit(parent_id="doc-1_p0000", distance=0.7),),
        search_rows=(make_hit(distance=dense_top),),
    )
    # 给一份带生效日期的材料：日期要来回经过一次 ISO 字符串做缓存，
    # 全空的话这条路径测不到
    mysql = make_mysql(
        documents=[make_document(effective_date=date(2021, 7, 1))]
    )
    embedding = FakeEmbeddingStore()
    client = FakeBailianClient(model_text)
    redis, redis_client = build_store()

    return Env(
        context=QaContext(
            mysql=mysql,
            milvus=milvus,
            embedding=embedding,
            client=client,
            redis=redis,
            config=RetrievalConfig(refuse_threshold=threshold),
        ),
        mysql=mysql,
        milvus=milvus,
        milvus_client=milvus_client,
        embedding=embedding,
        client=client,
        redis=redis_client,
    )


def ask(env: Env, **overrides: Any) -> Any:
    """按给定字段跑一次问答。"""
    fields: dict[str, Any] = {"question": QUESTION}
    fields.update(overrides)
    return answer(QaRequest(**fields), env.context)


class TestSingleTurn:
    def test_returns_an_answer(self) -> None:
        env = make_env()

        result = ask(env)

        assert "150 欧元" in result.answer.text

    def test_generates_a_session_id(self) -> None:
        # 不传会话 id 也要回一个，前端拿它开始一段新会话
        env = make_env()

        assert ask(env).session_id

    def test_uses_the_given_trace_id(self) -> None:
        env = make_env()

        result = answer(QaRequest(question=QUESTION), env.context, trace_id="t-1")

        assert result.trace_id == "t-1"

    def test_generates_a_trace_id_when_absent(self) -> None:
        env = make_env()

        assert ask(env).trace_id

    def test_reports_parent_count(self) -> None:
        env = make_env()

        assert ask(env).parent_count == 1

    def test_reports_the_quality_metric(self) -> None:
        env = make_env(dense_top=0.71)

        assert ask(env).top_score == 0.71


class TestCaching:
    def test_first_call_is_not_a_cache_hit(self) -> None:
        env = make_env()

        assert ask(env).cache_hit is False

    def test_second_call_hits_the_cache(self) -> None:
        env = make_env()
        ask(env)

        assert ask(env).cache_hit is True

    def test_cache_hit_does_not_call_the_model(self) -> None:
        # 命中缓存还调一次模型，等于缓存白做
        env = make_env()
        ask(env)
        calls_after_first = len(env.client.calls)

        ask(env)

        assert len(env.client.calls) == calls_after_first

    def test_cache_hit_keeps_the_answer(self) -> None:
        env = make_env()
        ask(env)

        assert "150 欧元" in ask(env).answer.text

    def test_cache_hit_keeps_the_citations(self) -> None:
        # 缓存命中的请求不会再检索，材料得从缓存里拿得出来，
        # 否则引用只剩一个 id，界面点开是空的
        env = make_env()
        ask(env)

        cited = ask(env).answer.citations.cited

        assert [parent.parent_id for parent in cited] == ["doc-1_p0000"]
        assert cited[0].text
        assert cited[0].matched_children

    def test_cache_hit_keeps_the_effective_date(self) -> None:
        # 日期要来回经过一次 ISO 字符串，转错了这一栏就没了
        env = make_env()
        ask(env)

        cited = ask(env).answer.citations.cited

        assert cited[0].effective_date == date(2021, 7, 1)

    def test_cache_hit_uses_a_fresh_session_id(self) -> None:
        # 缓存跨会话共享，回上一个会话的 id 会让前端把两段对话混在一起
        env = make_env()
        first = ask(env)

        second = ask(env)

        assert second.session_id != first.session_id

    def test_different_filters_do_not_share_the_cache(self) -> None:
        # 同一个问题在不同筛选下召回的材料不同，答案也可能不同
        env = make_env()
        ask(env, country="EU")

        assert ask(env, country="DE").cache_hit is False

    def test_same_filters_share_the_cache(self) -> None:
        env = make_env()
        ask(env, country="EU")

        assert ask(env, country="EU").cache_hit is True


class TestMultiTurn:
    def test_first_turn_is_not_a_cache_hit(self) -> None:
        env = make_env()

        assert ask(env, session_id="s1").cache_hit is False

    def test_multi_turn_does_not_use_the_cache(self) -> None:
        # 多轮的回答依赖前面的对话内容，用同一个键缓存会串
        env = make_env()
        ask(env, session_id="s1")

        assert ask(env, session_id="s1").cache_hit is False

    def test_multi_turn_does_not_write_the_cache(self) -> None:
        env = make_env()

        ask(env, session_id="s1")

        assert env.redis.strings == {}

    def test_uses_the_given_session_id(self) -> None:
        env = make_env()

        assert ask(env, session_id="s1").session_id == "s1"

    def test_records_both_messages(self) -> None:
        # 一轮问答是两条消息，前端与服务端都要按顺序看到
        env = make_env()

        ask(env, session_id="s1")

        messages = env.redis.lists["cbe:session:s1"]
        assert len(messages) == 2
        assert QUESTION in messages[0]
        assert "150 欧元" in messages[1]

    def test_refused_answer_is_also_recorded(self) -> None:
        # 使用者接着问「那到底是多少」时，两端都该知道上一轮答的是什么
        env = make_env(dense_top=0.1)

        result = ask(env, session_id="s1")

        assert result.refused is True
        assert len(env.redis.lists["cbe:session:s1"]) == 2


class TestTracing:
    def test_trace_id_is_a_uuid_when_generated(self) -> None:
        env = make_env()

        assert len(ask(env).trace_id) == 36


class TestRefusal:
    def test_below_threshold_is_refused(self) -> None:
        env = make_env(dense_top=0.1)

        assert ask(env).refused is True

    def test_refusal_does_not_call_the_model(self) -> None:
        env = make_env(dense_top=0.1)

        ask(env)

        assert env.client.calls == []

    def test_uncalibrated_threshold_raises(self) -> None:
        # 阈值没标定时不放行，这是配置里写死的约定
        env = make_env(threshold=None)

        try:
            ask(env)
        except UncalibratedThresholdError:
            return
        raise AssertionError("阈值未标定却没有报错")


class TestDegraded:
    def test_uncited_answer_is_degraded(self) -> None:
        env = make_env(model_text=UNCITED_ANSWER)

        assert ask(env).answer.degraded is True

    def test_degraded_answer_is_cached_too(self) -> None:
        # 降级是最终结果，缓存它比每次重跑一遍便宜
        env = make_env(model_text=UNCITED_ANSWER)
        ask(env)

        assert ask(env).cache_hit is True
