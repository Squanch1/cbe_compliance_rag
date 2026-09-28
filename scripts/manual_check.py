"""手动验证脚本：逐模块演示当前已完成的功能。

在 PyCharm 里右键本文件 → Run 即可，不需要任何命令行参数。

与 check_services.py 的区别：
    check_services.py  正式的连通性检查，有退出码，可接 CI
    manual_check.py    学习与排查用，逐段演示每个模块怎么调用

两个坑已在本文件内绕开：
    1. import 路径：把 src 加入搜索路径，不依赖 PyCharm 的 Sources Root 设置
    2. .env 路径：按本文件位置推算项目根目录，不依赖运行时的工作目录
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ENV_FILE = PROJECT_ROOT / ".env"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cbe_rag.observability import setup_console  # noqa: E402

# 让 print 的中文不乱码、且与日志的显示顺序一致
setup_console()

from cbe_rag.config.settings import Settings  # noqa: E402
from cbe_rag.observability import get_logger, setup_logging  # noqa: E402
from cbe_rag.storage import MilvusStore, MongoStore, MysqlStore, RedisStore  # noqa: E402
from cbe_rag.storage.health import StorageAdapter  # noqa: E402

LINE = "=" * 66


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def load_settings() -> Settings | None:
    """加载配置。失败时打印可读的提示并返回 None。"""
    section("1. 配置层 config/settings.py")

    if not ENV_FILE.is_file():
        print("  [FAIL] 找不到配置文件：%s" % ENV_FILE)
        print("         请复制 .env.example 为 .env 并填入真实凭据。")
        return None

    try:
        settings = Settings(_env_file=ENV_FILE)
    except Exception as exc:
        print("  [FAIL] 配置加载失败：%s" % type(exc).__name__)
        print("         %s" % str(exc)[:400])
        return None

    print("  [OK] 配置加载成功，来自 %s" % ENV_FILE.name)
    print()
    print("  Milvus   : %s:%d  库=%s" % (
        settings.milvus.host, settings.milvus.port, settings.milvus.database))
    print("  MySQL    : %s:%d  库=%s" % (
        settings.mysql.host, settings.mysql.port, settings.mysql.database))
    print("  MongoDB  : %s:%d  库=%s  认证库=%s" % (
        settings.mongodb.host, settings.mongodb.port,
        settings.mongodb.database, settings.mongodb.auth_source))
    print("  Redis    : %s:%d  键前缀=%s" % (
        settings.redis.host, settings.redis.port, settings.redis.key_prefix))
    print()
    print("  嵌入模型 : %s  fp16=%s  设备=%s" % (
        settings.embedding.model_path, settings.embedding.use_fp16,
        settings.embedding.device))
    print("  检索参数 : 稠密候选=%d 稀疏候选=%d 权重=%s/%s" % (
        settings.retrieval.dense_limit, settings.retrieval.sparse_limit,
        settings.retrieval.dense_weight, settings.retrieval.sparse_weight))
    print("  拒答阈值 : %r  （None 表示尚未标定）" % settings.retrieval.refuse_threshold)

    print()
    print("  密码保护检查（下面是 redis 与 llm 两段的 repr）：")
    print("    %s" % repr(settings.redis))
    print("    %s" % repr(settings.llm))
    print("  密码应显示为 SecretStr('**********')，不得出现明文。")

    return settings


def check_adapters(settings: Settings) -> None:
    section("2. 存储适配层 storage/  —— 统一遍历四个适配器")

    # 关键点：声明成 list[StorageAdapter] 之后，循环里不需要任何类型判断
    adapters: list[StorageAdapter] = [
        MilvusStore(settings.milvus),
        MongoStore(settings.mongodb),
        MysqlStore(settings.mysql),
        RedisStore(settings.redis),
    ]

    try:
        for adapter in adapters:
            print("  " + adapter.health_check().render())
    finally:
        for adapter in adapters:
            adapter.close()

    print()
    print("  上面这段代码里没有一行 if/else 判断具体是哪种适配器，")
    print("  这是 StorageAdapter 协议带来的好处。")


def check_single_adapter(settings: Settings) -> None:
    section("3. 单个适配器 —— 看返回值的结构")

    store = RedisStore(settings.redis)
    try:
        result = store.health_check()
        print("  service    = %s" % result.service)
        print("  ok         = %s" % result.ok)
        print("  detail     = %s" % result.detail)
        print("  elapsed_ms = %.2f" % result.elapsed_ms)
        print("  render()   = %s" % result.render())
    finally:
        store.close()

    print()
    print("  失败时 ok=False，detail 里是异常类型与消息，不会抛异常出来。")


def check_logging() -> None:
    section("4. 日志 observability/logging.py")

    setup_logging("DEBUG")
    logger = get_logger("cbe_rag.manual_check")

    print("  下面四条是演示文本，用来看格式和级别过滤效果。")
    print("  内容本身没有含义，不代表任何真实服务状态。")
    print()
    logger.debug("【示例】调试级别，只有 setup_logging(DEBUG) 时才显示")
    logger.info("【示例】信息级别")
    logger.warning("【示例】警告级别，参数会被填进占位符：%s", "这里")
    logger.error("【示例】错误级别")

    print()
    print("  再看错误处理：传一个不存在的级别名")
    try:
        setup_logging("VERBOSE")
        print("  [问题] 没有报错，说明静默降级了")
    except ValueError as exc:
        print("  [OK] 按预期报错：%s" % exc)


def main() -> int:
    print()
    print("已完成模块的手动验证")

    settings = load_settings()
    if settings is None:
        return 1

    check_adapters(settings)
    check_single_adapter(settings)
    check_logging()

    print()
    print(LINE)
    print("验证结束。若有 FAIL 项，先解决再往下走。")
    print(LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
