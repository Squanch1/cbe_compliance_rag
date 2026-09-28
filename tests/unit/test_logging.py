"""日志工具的单元测试。"""

from __future__ import annotations

import logging
from typing import Iterator, TextIO

import pytest

from cbe_rag.observability.logging import (
    get_logger,
    open_utf8_stream,
    setup_console,
    setup_logging,
)


class RecordingHandler(logging.Handler):
    """把收到的日志消息收集起来，便于断言。"""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def clean_root() -> Iterator[None]:
    """备份并还原根日志器状态，避免测试之间互相干扰。"""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def attach_recorder() -> RecordingHandler:
    """在根日志器上挂一个记录器，用于收集日志消息。

    必须在 setup_logging() 之后调用。setup_logging 会清空根日志器上
    已有的 handler，在此之前挂上的记录器会被一并移除，导致断言拿到空列表。
    """
    handler = RecordingHandler()
    logging.getLogger().addHandler(handler)
    return handler


class TestGetLogger:
    def test_returns_logger_with_given_name(self) -> None:
        assert get_logger("cbe_rag.demo").name == "cbe_rag.demo"

    def test_does_not_configure_root(self, clean_root: None) -> None:
        # 库代码取 logger 不应产生副作用，否则使用方的日志设置会被悄悄改掉
        root = logging.getLogger()
        handlers_before = list(root.handlers)
        level_before = root.level

        get_logger("cbe_rag.demo")

        assert root.handlers == handlers_before
        assert root.level == level_before


class TestSetupLogging:
    def test_adds_exactly_one_handler(self, clean_root: None) -> None:
        setup_logging()

        assert len(logging.getLogger().handlers) == 1

    def test_repeated_calls_do_not_stack_handlers(self, clean_root: None) -> None:
        # 重复初始化（重启生命周期、重跑入口）不应让同一条日志打印多次
        setup_logging()
        setup_logging()
        setup_logging()

        assert len(logging.getLogger().handlers) == 1

    def test_invalid_level_raises(self, clean_root: None) -> None:
        # 级别名写错时不能静默退回默认级别，否则调了 DEBUG 却只看到 INFO，
        # 且完全找不到原因
        with pytest.raises(ValueError):
            setup_logging("VERBOSE")

    def test_accepts_lowercase_level(self, clean_root: None) -> None:
        setup_logging("debug")

        assert logging.getLogger().level == logging.DEBUG

    def test_accepts_integer_level(self, clean_root: None) -> None:
        setup_logging(logging.WARNING)

        assert logging.getLogger().level == logging.WARNING


class TestLevelFiltering:
    def test_info_level_drops_debug(self, clean_root: None) -> None:
        setup_logging("INFO")
        recorder = attach_recorder()
        logger = get_logger("cbe_rag.demo")

        logger.debug("不该出现")
        logger.info("应该出现")

        assert recorder.messages == ["应该出现"]

    def test_debug_level_keeps_debug(self, clean_root: None) -> None:
        setup_logging("DEBUG")
        recorder = attach_recorder()
        logger = get_logger("cbe_rag.demo")

        logger.debug("应该出现")

        assert recorder.messages == ["应该出现"]

    def test_children_of_configured_logger_inherit_level(self, clean_root: None) -> None:
        setup_logging("WARNING")
        recorder = attach_recorder()
        logger = get_logger("cbe_rag.storage.mysql_store")

        logger.info("不该出现")
        logger.warning("应该出现")

        assert recorder.messages == ["应该出现"]


class TestChineseMessages:
    def test_message_survives_unchanged(self, clean_root: None) -> None:
        # 失败详情里带中文（例如 Milvus 报「未创建，需先初始化」），
        # 日志层不应改变它
        setup_logging("INFO")
        recorder = attach_recorder()

        get_logger("cbe_rag.demo").warning("数据库 %s", "未创建，需先初始化")

        assert recorder.messages == ["数据库 未创建，需先初始化"]


class TestUtf8Stream:
    def test_returns_stream_without_reconfigure_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 某些环境下 sys.stderr 已被替换成不支持 reconfigure 的对象
        class PlainStream:
            def write(self, text: str) -> int:
                return len(text)

            def flush(self) -> None:
                pass

        plain = PlainStream()
        monkeypatch.setattr("sys.stderr", plain)

        assert open_utf8_stream() is plain

    def test_reconfigure_failure_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenStream:
            def reconfigure(self, **kwargs: object) -> None:
                raise ValueError("stream already detached")

        broken = BrokenStream()
        monkeypatch.setattr("sys.stderr", broken)

        # 流已分离等情况下应保持原样，不能阻断程序启动
        assert open_utf8_stream() is broken

    def test_reconfigures_to_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class RecordingStream:
            def reconfigure(self, **kwargs: object) -> None:
                captured.update(kwargs)

        stream = RecordingStream()
        monkeypatch.setattr("sys.stderr", stream)

        open_utf8_stream()

        assert captured["encoding"] == "utf-8"
        assert captured["errors"] == "replace"

    def test_stderr_does_not_force_line_buffering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 标准错误本就不缓冲，显式传 line_buffering=False 反而可能改坏它
        captured: dict[str, object] = {}

        class RecordingStream:
            def reconfigure(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("sys.stderr", RecordingStream())

        open_utf8_stream()

        assert "line_buffering" not in captured


class TestSetupConsole:
    """标准输出的配置。

    解决两个问题：PyCharm 控制台与管道环境下中文乱码；
    标准输出块缓冲导致与日志的显示顺序错乱。
    """

    def test_reconfigures_stdout_to_utf8_with_line_buffering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        class RecordingStream:
            def reconfigure(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("sys.stdout", RecordingStream())

        setup_console()

        assert captured["encoding"] == "utf-8"
        assert captured["line_buffering"] is True

    def test_stream_without_reconfigure_is_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class PlainStream:
            pass

        monkeypatch.setattr("sys.stdout", PlainStream())

        setup_console()

    def test_reconfigure_failure_does_not_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BrokenStream:
            def reconfigure(self, **kwargs: object) -> None:
                raise ValueError("stream already detached")

        monkeypatch.setattr("sys.stdout", BrokenStream())

        setup_console()
