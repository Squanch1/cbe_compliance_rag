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


def open_utf8_stream() -> TextIO:
    """返回一个能正确输出中文的标准错误流。

    Windows 上标准错误被重定向到管道时，Python 会退回本地编码
    （本机为 cp936），中文会变成乱码。这里显式改成 UTF-8。

    errors="replace" 是兜底：遇到编码不了的字符时替换而不是抛异常，
    日志本身不应该成为程序崩溃的原因。
    """
    stream = sys.stderr
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 流已被分离或已关闭等情况下保持原样，不阻断启动
            pass
    return stream


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
