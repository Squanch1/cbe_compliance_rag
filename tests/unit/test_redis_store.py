"""Redis 适配器的单元测试。

注入假客户端，不连接真实服务。
"""

from __future__ import annotations

import json

import pytest
from redis.exceptions import AuthenticationError, ConnectionError as RedisConnectionError

from cbe_rag.storage.redis_store import RedisStore
from redis_fakes import FakeRedis, build_store, make_config


class TestSessionStorage:
    def test_appends_and_reads_back_in_order(self) -> None:
        # 一轮问答是两条消息，要按顺序整体读回来
        store, _ = build_store()

        store.append_message("s1", {"role": "user", "content": "问题"})
        store.append_message("s1", {"role": "assistant", "content": "回答"})

        assert [item["role"] for item in store.load_session("s1")] == [
            "user",
            "assistant",
        ]

    def test_key_carries_the_prefix(self) -> None:
        # 实例由多个项目共用，不加前缀会和其他项目的键混在一起
        store, client = build_store()

        store.append_message("s1", {"role": "user", "content": "问题"})

        assert list(client.lists) == ["cbe:session:s1"]

    def test_sessions_are_isolated(self) -> None:
        store, _ = build_store()

        store.append_message("s1", {"role": "user", "content": "一"})
        store.append_message("s2", {"role": "user", "content": "二"})

        assert store.load_session("s1")[0]["content"] == "一"
        assert store.load_session("s2")[0]["content"] == "二"

    def test_missing_session_is_empty(self) -> None:
        store, _ = build_store()

        assert store.load_session("nope") == []

    def test_ttl_is_applied(self) -> None:
        store, client = build_store()

        store.append_message("s1", {"role": "user", "content": "问题"})

        assert client.expirations["cbe:session:s1"] == 3600

    def test_ttl_is_refreshed_on_every_append(self) -> None:
        # 会话过期时间要随每次使用续期，否则长对话会在中途断掉
        store, client = build_store()

        store.append_message("s1", {"role": "user", "content": "一"})
        client.expirations.clear()
        store.append_message("s1", {"role": "user", "content": "二"})

        assert client.expirations["cbe:session:s1"] == 3600

    def test_old_messages_are_trimmed(self) -> None:
        # 上下文长度直接影响每次调用的成本
        store, _ = build_store(max_messages=3)

        for index in range(5):
            store.append_message("s1", {"role": "user", "content": str(index)})

        assert [item["content"] for item in store.load_session("s1")] == [
            "2",
            "3",
            "4",
        ]

    def test_trimming_keeps_the_latest(self) -> None:
        # 最早的几轮对当前问题参考价值最低，该丢的是它们
        store, _ = build_store(max_messages=1)

        store.append_message("s1", {"role": "user", "content": "旧"})
        store.append_message("s1", {"role": "user", "content": "新"})

        assert [item["content"] for item in store.load_session("s1")] == ["新"]

    def test_newlines_survive_the_round_trip(self) -> None:
        store, _ = build_store()

        store.append_message("s1", {"role": "user", "content": "第一行\n第二行"})

        assert store.load_session("s1")[0]["content"] == "第一行\n第二行"

    def test_unicode_survives_the_round_trip(self) -> None:
        store, _ = build_store()

        store.append_message("s1", {"role": "user", "content": "进口一站式服务"})

        assert store.load_session("s1")[0]["content"] == "进口一站式服务"


class TestAnswerCache:
    def test_round_trip(self) -> None:
        store, _ = build_store()

        store.cache_answer("abc", {"answer": "150 欧元"})

        assert store.get_cached_answer("abc") == {"answer": "150 欧元"}

    def test_missing_returns_none(self) -> None:
        store, _ = build_store()

        assert store.get_cached_answer("nope") is None

    def test_key_carries_the_prefix(self) -> None:
        store, client = build_store()

        store.cache_answer("abc", {"answer": "x"})

        assert list(client.strings) == ["cbe:cache:abc"]

    def test_cache_ttl_is_shorter_than_session_ttl(self) -> None:
        # 检索结果会随语料变化，缓存该比会话先过期
        store, client = build_store()

        store.cache_answer("abc", {"answer": "x"})
        store.append_message("s1", {"role": "user", "content": "问"})

        assert (
            client.expirations["cbe:cache:abc"]
            < client.expirations["cbe:session:s1"]
        )

    def test_stored_value_is_plain_json(self) -> None:
        # spec 03 要求：Redis 里的数据可能跨进程跨语言读取，
        # 不用 pickle 或任何可执行序列化格式
        store, client = build_store()

        store.cache_answer("abc", {"answer": "进口一站式服务"})

        raw = client.strings["cbe:cache:abc"]
        assert json.loads(raw) == {"answer": "进口一站式服务"}
        # ensure_ascii=False：中文原样存，不转成 \uXXXX，便于直接看
        assert "进口一站式服务" in raw


class TestFailurePropagates:
    def test_session_read_failure_raises(self) -> None:
        # 会话与缓存不是「报告状态」类接口，出错要让调用方知道
        store, _ = build_store(fail_with=RedisConnectionError("连接断了"))

        with pytest.raises(RedisConnectionError):
            store.load_session("s1")

    def test_cache_write_failure_raises(self) -> None:
        store, _ = build_store(fail_with=RedisConnectionError("连接断了"))

        with pytest.raises(RedisConnectionError):
            store.cache_answer("abc", {"answer": "x"})


class TestHealthCheckSuccess:
    def test_reports_ok_and_version(self) -> None:
        client = FakeRedis(version="7.2.4")
        store = RedisStore(make_config(), client=client)

        result = store.health_check()

        assert result.ok is True
        assert result.service == "redis"
        assert "7.2.4" in result.detail

    def test_ping_is_actually_called(self) -> None:
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.health_check()

        assert client.ping_called == 1

    def test_elapsed_time_is_recorded(self) -> None:
        store = RedisStore(make_config(), client=FakeRedis())

        result = store.health_check()

        assert result.elapsed_ms >= 0.0

    def test_missing_version_field_does_not_crash(self) -> None:
        # 某些托管 Redis 的 info("server") 不返回 redis_version
        store = RedisStore(make_config(), client=FakeRedis(info_missing_version=True))

        result = store.health_check()

        assert result.ok is True
        assert "unknown" in result.detail


class TestHealthCheckFailure:
    @pytest.mark.parametrize(
        "exc",
        [
            AuthenticationError("Authentication required."),
            RedisConnectionError("Connection refused"),
            OSError("timed out"),
        ],
    )
    def test_failure_is_reported_not_raised(self, exc: Exception) -> None:
        # 健康检查的职责是报告问题，不是把异常抛给调用方
        store = RedisStore(make_config(), client=FakeRedis(fail_with=exc))

        result = store.health_check()

        assert result.ok is False

    def test_failure_detail_names_the_exception_type(self) -> None:
        exc = AuthenticationError("Authentication required.")
        store = RedisStore(make_config(), client=FakeRedis(fail_with=exc))

        result = store.health_check()

        assert "AuthenticationError" in result.detail
        assert "Authentication required." in result.detail

    def test_unexpected_exception_is_reported_not_raised(self) -> None:
        # 回归：失败来源不止驱动异常，任何异常都不得穿透 health_check
        store = RedisStore(make_config(), client=FakeRedis(fail_with=ValueError("boom")))

        result = store.health_check()

        assert result.ok is False
        assert "ValueError" in result.detail

    def test_password_is_not_leaked_in_failure_detail(self) -> None:
        config = make_config()
        store = RedisStore(
            config,
            client=FakeRedis(fail_with=RedisConnectionError("Connection refused")),
        )

        result = store.health_check()

        assert config.password.get_secret_value() not in result.detail


class TestLifecycle:
    def test_health_check_does_not_close_the_client(self) -> None:
        # redis-py 内部维护连接池，客户端应当长期复用
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.health_check()

        assert client.closed is False

    def test_close_releases_the_client(self) -> None:
        client = FakeRedis()
        store = RedisStore(make_config(), client=client)

        store.close()

        assert client.closed is True


class TestHealthResultRendering:
    def test_ok_line_contains_service_and_timing(self) -> None:
        store = RedisStore(make_config(), client=FakeRedis(version="7.2.4"))

        line = store.health_check().render()

        assert line.startswith("[OK")
        assert "redis" in line
        assert "7.2.4" in line

    def test_fail_line_is_marked(self) -> None:
        store = RedisStore(
            make_config(),
            client=FakeRedis(fail_with=RedisConnectionError("Connection refused")),
        )

        line = store.health_check().render()

        assert line.startswith("[FAIL]")
