"""外部服务与本地模型连通性检查。

用法：
    python scripts/check_services.py

退出码：
    0  全部通过
    1  配置加载失败，或存在未通过项

探测逻辑都在 storage/ 的适配器里，本脚本只负责构造、遍历与汇总。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接以脚本方式运行时，把 src 加入模块搜索路径。
# 通过 pytest 运行时由 pyproject.toml 的 pythonpath 配置提供，此处不生效。
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import ENV_EXAMPLE_PATH, ENV_FILE_PATH, Settings  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.storage import (  # noqa: E402
    EmbeddingStore,
    MilvusStore,
    MongoStore,
    MysqlStore,
    RedisStore,
)
from cbe_rag.storage.health import StorageAdapter  # noqa: E402


def build_adapters(settings: Settings) -> list[StorageAdapter]:
    """构造全部需要检查的组件。

    四个外部服务加一个进程内加载的模型。五者都实现 StorageAdapter 协议，
    因此下面可以不加判断地统一遍历。
    """
    return [
        MilvusStore(settings.milvus),
        MongoStore(settings.mongodb),
        MysqlStore(settings.mysql),
        RedisStore(settings.redis),
        EmbeddingStore(settings.embedding),
    ]


def report_config_error(exc: ValidationError) -> None:
    """打印配置错误。

    只输出字段路径与错误类型，不输出错误里附带的输入值——
    pydantic 的报错会带上完整的输入字典，其中包含密码。
    """
    print("[FAIL] 配置加载失败，以下配置项缺失或写错：")
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        print("       - %-32s %s" % (location, error["msg"]))
    print()
    print("       请对照 .env.example 检查 .env。")


def main() -> int:
    """执行全部检查并返回退出码。"""
    # 不改的话，PyCharm 控制台与管道环境下本脚本的中文报告会乱码
    setup_console()

    print("=" * 64)
    print("服务连通性检查")
    print("=" * 64)

    # 先单独判文件在不在。否则 Settings 会把「文件没找到」报成
    # 一堆「字段缺失」，让人跑去检查 .env 的内容，而问题其实在路径上。
    if not ENV_FILE_PATH.is_file():
        print("[FAIL] 找不到配置文件：%s" % ENV_FILE_PATH)
        print()
        print("       请先复制配置模板并填入真实凭据：")
        print("         copy %s %s" % (ENV_EXAMPLE_PATH.name, ENV_FILE_PATH.name))
        return 1

    try:
        settings = Settings()
    except ValidationError as exc:
        report_config_error(exc)
        return 1

    adapters = build_adapters(settings)
    try:
        results = [adapter.health_check() for adapter in adapters]
    finally:
        for adapter in adapters:
            adapter.close()

    print()
    for result in results:
        print("  " + result.render())

    failed = [result for result in results if not result.ok]
    print()
    if failed:
        print("未通过 %d 项，共 %d 项：" % (len(failed), len(results)))
        for result in failed:
            print("  - %s：%s" % (result.service, result.detail))
        return 1

    print("全部 %d 项通过。" % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
