"""日志配置。

用法约定：

- 各模块在文件顶部取 logger：``logger = get_logger(__name__)``
- 只在程序入口调用一次 ``setup_logging()``

库代码只取 logger、不配置 logger。配置是入口的职责，
否则被 import 时就会悄悄改掉使用方的日志设置。
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

DEFAULT_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """返回指定名字的 logger。

    只是 logging.getLogger 的薄封装。保留这一层是为了统一入口，
    将来若更换日志后端，改动集中在此处。
    """
    return logging.getLogger(name)


def _reconfigure_utf8(stream: TextIO, *, line_buffering: bool = False) -> TextIO:
    """把流改成 UTF-8 输出。

    errors="replace" 是兜底：遇到编码不了的字符时替换而不是抛异常，
    输出本身不应该成为程序崩溃的原因。

    流不支持 reconfigure（已被替换成自定义对象）或已被分离时保持原样，
    不阻断启动。
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream

    kwargs: dict[str, object] = {"encoding": "utf-8", "errors": "replace"}
    if line_buffering:
        # 只在需要打开时传入，避免把本就开启行缓冲的流关掉
        kwargs["line_buffering"] = True

    try:
        reconfigure(**kwargs)
    except (ValueError, OSError):
        pass
    return stream


def open_utf8_stream() -> TextIO:
    """返回一个能正确输出中文的标准错误流。

    Windows 上标准错误被重定向到管道时，Python 会退回本地编码
    （本机为 cp936），中文会变成乱码。这里显式改成 UTF-8。
    """
    return _reconfigure_utf8(sys.stderr)


def setup_console() -> None:
    """配置标准输出。脚本入口调用一次。

    解决两个问题：

    编码——PyCharm 控制台与管道环境下标准输出用本地编码（本机 cp936），
    print 出来的中文会乱码。日志不受影响，因为 setup_logging 已经把
    标准错误改成 UTF-8 了。

    缓冲——标准输出在非终端环境下是块缓冲，而标准错误不缓冲。两者混用时
    显示顺序会与执行顺序不一致：报告明明先打印，却显示在日志后面。
    改成行缓冲即可对齐。
    """
    _reconfigure_utf8(sys.stdout, line_buffering=True)


def setup_logging(level: str | int = "INFO") -> None:
    """配置根日志器。只在程序入口调用。

    重复调用会替换已有 handler，不会叠加——否则每次重启生命周期
    或重跑初始化都会让同一条日志打印多次。

    级别名写错时抛 ValueError 而不是退回默认级别。
    静默降级会让人以为调了 DEBUG 却只看到 INFO，且找不到原因。
    """
    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        if not isinstance(resolved, int):
            raise ValueError("无效的日志级别：%r" % level)
    else:
        resolved = level

    handler = logging.StreamHandler(open_utf8_stream())
    handler.setFormatter(
        logging.Formatter(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)
