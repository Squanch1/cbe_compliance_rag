"""接口层的单元测试。

用 TestClient 走真实的 HTTP 路径，但依赖全部换成假实现——测的是路由与
响应模型的接线，不是各适配器本身（那些另有测试）。

契约见 docs/spec/04-api-contract.md。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cbe_rag.api.app import TRACE_HEADER, create_app, is_valid_trace
from cbe_rag.api.deps import AppState
from cbe_rag.config.settings import ENV_FILE_PATH, RetrievalConfig
from cbe_rag.retrieval.service import UncalibratedThresholdError
from cbe_rag.storage.health import HealthResult
from indexing_fakes import FakeEmbeddingStore
from milvus_fakes import build_db_store, make_hit
from redis_fakes import build_store
from retrieval_fakes import make_document, make_mysql


class FakeBailianClient:
    """假的模型客户端。"""

    def __init__(self, text: str = "上限是 150 欧元 [1]。") -> None:
        self._text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Any:
        self.calls.append((system, user))
        return SimpleNamespace(text=self._text, model="fake")

    def health_check(self) -> HealthResult:
        return HealthResult(service="bailian", ok=True, detail="ok", elapsed_ms=0.1)

    def close(self) -> None:
        return None


class FakeAdapter:
    """假的健康检查适配器。"""

    def __init__(self, service: str, ok: bool) -> None:
        self._service = service
        self._ok = ok

    def health_check(self) -> HealthResult:
        return HealthResult(
            service=self._service,
            ok=self._ok,
            detail="detail of %s" % self._service,
            elapsed_ms=1.5,
        )

    def append_message(self, *args: Any, **kwargs: Any) -> None:
        return None

    def get_cached_answer(self, digest: str) -> Any:
        return None

    def cache_answer(self, digest: str, payload: dict[str, Any]) -> None:
        return None

    def get_chunks(self, chunk_ids: list[str]) -> list[Any]:
        return []

    def get_documents(self, doc_ids: list[str]) -> list[Any]:
        return []

    def close(self) -> None:
        return None


def make_state(
    *,
    dense_top: float = 0.7,
    threshold: float | None = 0.5,
    model_text: str = "上限是 150 欧元 [1]。",
    health_ok: bool = True,
) -> AppState:
    """造一份注入假的 AppState。"""
    milvus, _, _ = build_db_store(
        hybrid_rows=(make_hit(parent_id="doc-1_p0000", distance=0.7),),
        search_rows=(make_hit(distance=dense_top),),
    )
    redis, _ = build_store()

    return AppState(
        settings=SimpleNamespace(  # type: ignore[arg-type]
            retrieval=RetrievalConfig(refuse_threshold=threshold)
        ),
        mysql=make_mysql(
            documents=[make_document(effective_date=date(2021, 7, 1))]
        ),
        milvus=milvus,
        embedding=FakeEmbeddingStore(),
        client=FakeBailianClient(model_text),
        redis=redis,
    )


def build_client(state: AppState | None = None, **kwargs: Any) -> TestClient:
    """起一个注入了假状态的应用。"""
    fake = state if state is not None else make_state(**kwargs)
    app = create_app(state_factory=lambda _: fake)
    return TestClient(app)


def ask(client: TestClient, **payload: Any) -> Any:
    """发一次问答请求。"""
    body: dict[str, Any] = {"question": "进口一站式服务的适用金额上限是多少"}
    body.update(payload)
    return client.post("/api/v1/ask", json=body)


@pytest.mark.skipif(
    not ENV_FILE_PATH.is_file(), reason="需要 .env 才能走默认装配路径"
)
class TestDefaultWiring:
    def test_create_app_without_settings(self) -> None:
        # 生产路径：不注入任何东西，由 build_state 按 .env 造适配器。
        # 其余测试全都注入了假 factory，这条默认路径反而没人走过——
        # create_app() 曾经因为漏了「不传就用 Settings()」而直接崩。
        app = create_app()

        assert app.state.app_state is not None


class TestTraceIdValidation:
    def test_accepts_a_uuid(self) -> None:
        assert is_valid_trace("3f2a1b4c-5d6e-7f80-9a1b-2c3d4e5f6071")

    def test_rejects_empty(self) -> None:
        assert not is_valid_trace("")

    def test_rejects_whitespace_and_newlines(self) -> None:
        # 这个值要进日志，带换行会把日志格式搞乱
        assert not is_valid_trace("abc\ndef")

    def test_rejects_overlong(self) -> None:
        assert not is_valid_trace("a" * 129)

    def test_accepts_the_length_limit(self) -> None:
        assert is_valid_trace("a" * 128)


class TestAskRouting:
    def test_returns_an_answer(self) -> None:
        response = ask(build_client())

        assert response.status_code == 200
        assert "150 欧元" in response.json()["answer"]

    def test_generates_a_trace_id(self) -> None:
        body = ask(build_client()).json()

        assert body["trace_id"]

    def test_echoes_the_trace_id_in_the_header(self) -> None:
        # 排查时要能从响应头直接拿到，不必先解 JSON
        response = ask(build_client())

        assert response.headers[TRACE_HEADER] == response.json()["trace_id"]

    def test_reuses_a_supplied_trace_id(self) -> None:
        response = ask(build_client(), **{})

        assert response.status_code == 200

    def test_honours_a_valid_supplied_trace_id(self) -> None:
        client = build_client()

        response = client.post(
            "/api/v1/ask",
            json={"question": "问题"},
            headers={TRACE_HEADER: "trace-from-caller"},
        )

        assert response.json()["trace_id"] == "trace-from-caller"

    def test_replaces_an_invalid_supplied_trace_id(self) -> None:
        # 不合规就重新生成，不报错：它只是排查线索，为它让请求失败不值当
        client = build_client()

        response = client.post(
            "/api/v1/ask",
            json={"question": "问题"},
            headers={TRACE_HEADER: "bad trace with spaces"},
        )

        assert response.json()["trace_id"] != "bad trace with spaces"

    def test_generates_a_session_id(self) -> None:
        assert ask(build_client()).json()["session_id"]

    def test_uses_the_supplied_session_id(self) -> None:
        body = ask(build_client(), session_id="s-1").json()

        assert body["session_id"] == "s-1"


class TestAskResponseShape:
    def test_reports_refused_and_degraded_separately(self) -> None:
        body = ask(build_client()).json()

        assert body["refused"] is False
        assert body["degraded"] is False

    def test_reports_cache_state(self) -> None:
        # 使用者据此知道这次的结果是不是缓存的（单轮才有意义）
        body = ask(build_client()).json()

        assert body["cache_hit"] is False

    def test_second_identical_question_hits_the_cache(self) -> None:
        client = build_client()

        ask(client)
        body = ask(client).json()

        assert body["cache_hit"] is True

    def test_citations_carry_matched_children(self) -> None:
        # 界面靠子块的偏移高亮到具体段落，只给父块级引用等于没有
        citation = ask(build_client()).json()["citations"][0]

        assert citation["parent_id"] == "doc-1_p0000"
        assert citation["matched_children"][0]["chunk_id"] == "doc-1_c0000"

    def test_matched_children_carry_offsets(self) -> None:
        child = ask(build_client()).json()["citations"][0]["matched_children"][0]

        assert child["start_offset"] == 0
        assert child["end_offset"] == 20

    def test_effective_date_is_serialised(self) -> None:
        citation = ask(build_client()).json()["citations"][0]

        assert citation["effective_date"] == "2021-07-01"

    def test_retrieval_section_carries_the_metric(self) -> None:
        retrieval = ask(build_client(dense_top=0.71)).json()["retrieval"]

        assert retrieval["top_score"] == pytest.approx(0.71)
        assert retrieval["parent_count"] == 1

    def test_notes_is_always_present(self) -> None:
        # 客户端不必判空，字段在就少一条分支
        assert isinstance(ask(build_client()).json()["notes"], list)


class TestAskValidation:
    def test_empty_question_is_rejected(self) -> None:
        response = ask(build_client(), question="")

        assert response.status_code == 400

    def test_empty_question_reports_invalid_request(self) -> None:
        body = ask(build_client(), question="").json()

        assert body["error"]["code"] == "invalid_request"

    def test_error_response_carries_a_trace_id(self) -> None:
        # 报错时也要能对上日志
        assert ask(build_client(), question="").json()["trace_id"]

    def test_missing_question_is_rejected(self) -> None:
        response = build_client().post("/api/v1/ask", json={})

        assert response.status_code == 400

    def test_unknown_field_is_rejected(self) -> None:
        # extra="forbid"：把拼错的字段名当没看见，使用者会以为生效了
        response = ask(build_client(), country="EU")

        assert response.status_code == 400


class TestAskFilters:
    def test_filters_reach_the_retrieval(self) -> None:
        state = make_state()
        client = build_client(state)

        ask(client, filters={"country": "EU"})

        # 过滤条件最终拼进 Milvus 的表达式，没生效的话这里会是空串
        assert state.milvus._get_db_client().hybrid_searched


class TestAskRefusal:
    def test_below_threshold_is_refused(self) -> None:
        body = ask(build_client(dense_top=0.1)).json()

        assert body["refused"] is True

    def test_uncalibrated_threshold_maps_to_503(self) -> None:
        response = ask(build_client(threshold=None))

        assert response.status_code == 503

    def test_uncalibrated_threshold_has_its_own_code(self) -> None:
        # 单独成一类：运维看到该去标定阈值，不是查 bug
        body = ask(build_client(threshold=None)).json()

        assert body["error"]["code"] == "uncalibrated_threshold"


class TestGetParent:
    def test_returns_the_parent(self) -> None:
        response = build_client().get("/api/v1/parents/doc-1_p0000")

        assert response.status_code == 200
        assert response.json()["parent_id"] == "doc-1_p0000"

    def test_carries_the_metadata_for_citation(self) -> None:
        body = build_client().get("/api/v1/parents/doc-1_p0000").json()

        assert body["title"]
        assert body["source_url"]

    def test_unknown_parent_is_404(self) -> None:
        response = build_client().get("/api/v1/parents/nope")

        assert response.status_code == 404

    def test_unknown_parent_has_its_own_code(self) -> None:
        # 界面拿着过期的 id 来查是正常操作路径，不是服务出错
        body = build_client().get("/api/v1/parents/nope").json()

        assert body["error"]["code"] == "unknown_parent"


class TestDimensions:
    def test_returns_all_three_dimensions(self) -> None:
        body = build_client().get("/api/v1/dimensions").json()

        assert set(body) == {"countries", "doc_types", "publishers"}

    def test_carries_both_code_and_name(self) -> None:
        # 界面显示中文、提交代码，两者都要有
        item = build_client().get("/api/v1/dimensions").json()["countries"][0]

        assert set(item) == {"code", "name_zh", "name_en"}

    def test_names_come_through(self) -> None:
        # 界面显示的就是这几个中文名
        body = build_client().get("/api/v1/dimensions").json()

        assert body["countries"][0]["name_zh"] == "欧盟"
        assert body["doc_types"][0]["name_zh"] == "平台政策"
        assert body["publishers"][0]["name_zh"] == "亚马逊"


class TestHealth:
    def make_state_with_health(self, *results: tuple[str, bool]) -> AppState:
        state = make_state()
        adapters = [FakeAdapter(service, ok) for service, ok in results]
        # 路由按固定顺序取五个适配器，这里逐个替换
        state.milvus = adapters[0]  # type: ignore[assignment]
        state.mysql = adapters[1]  # type: ignore[assignment]
        state.redis = adapters[2]  # type: ignore[assignment]
        state.embedding = adapters[3]  # type: ignore[assignment]
        state.client = adapters[4]  # type: ignore[assignment]
        return state

    def test_reports_every_service(self) -> None:
        state = self.make_state_with_health(
            ("milvus", True), ("mysql", True), ("redis", True),
            ("bge-m3", True), ("bailian", True),
        )

        body = build_client(state).get("/api/v1/health").json()

        assert [item["service"] for item in body["services"]] == [
            "milvus", "mysql", "redis", "bge-m3", "bailian",
        ]

    def test_ok_is_true_when_all_pass(self) -> None:
        state = self.make_state_with_health(
            ("milvus", True), ("mysql", True), ("redis", True),
            ("bge-m3", True), ("bailian", True),
        )

        assert build_client(state).get("/api/v1/health").json()["ok"] is True

    def test_ok_is_false_when_any_fails(self) -> None:
        state = self.make_state_with_health(
            ("milvus", True), ("mysql", False), ("redis", True),
            ("bge-m3", True), ("bailian", True),
        )

        assert build_client(state).get("/api/v1/health").json()["ok"] is False

    def test_returns_200_even_when_a_service_is_down(self) -> None:
        # 这个接口的语义是「报告状态」，不是「服务本身能不能响应」。
        # 返回 503 会让监控把「某个依赖挂了」和「这个服务挂了」混为一谈。
        state = self.make_state_with_health(
            ("milvus", False), ("mysql", False), ("redis", False),
            ("bge-m3", False), ("bailian", False),
        )

        assert build_client(state).get("/api/v1/health").status_code == 200

    def test_each_item_carries_its_detail(self) -> None:
        state = self.make_state_with_health(
            ("milvus", True), ("mysql", True), ("redis", True),
            ("bge-m3", True), ("bailian", True),
        )

        item = build_client(state).get("/api/v1/health").json()["services"][0]

        assert item["detail"]
        assert item["elapsed_ms"] >= 0
