"""手动验证脚本：真实调用已完成的模块，检查外部服务是否连通。

在 PyCharm 里右键本文件 → Run 即可，不需要任何命令行参数。

本脚本不含任何模拟数据——每一行输出都由真实的函数调用产生。
不通过的项目会如实报出来，不会假装成功。

与 check_services.py 的区别：
    check_services.py  正式的环境检查，输出精简，有退出码，可接 CI
    manual_check.py    排查与学习用，逐段展开，能看到返回值内部结构

两个环境坑已在本文件内绕开：
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

from cbe_rag.config.settings import Settings  # noqa: E402
from cbe_rag.observability import get_logger, setup_console, setup_logging  # noqa: E402
from cbe_rag.storage import MilvusStore, MongoStore, MysqlStore, RedisStore  # noqa: E402
from cbe_rag.storage.health import HealthResult, StorageAdapter  # noqa: E402

# 让 print 的中文不乱码，且与日志的显示顺序一致
setup_console()

LINE = "=" * 66


def section(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def build_adapters(settings: Settings) -> list[StorageAdapter]:
    """按配置构造四个存储适配器。

    声明成 list[StorageAdapter] 之后，下面的循环不需要任何类型判断——
    这是 StorageAdapter 协议带来的好处。
    """
    return [
        MilvusStore(settings.milvus),
        MongoStore(settings.mongodb),
        MysqlStore(settings.mysql),
        RedisStore(settings.redis),
    ]


def show_config() -> Settings | None:
    """加载并打印配置。配置有问题时返回 None。"""
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
    print("  密码保护：下面两行是真实配置对象的 repr，密码应为掩码")
    print("    %s" % repr(settings.redis))
    print("    %s" % repr(settings.llm))

    return settings


def probe_services(settings: Settings) -> list[HealthResult]:
    """真实调用四个服务的 health_check，返回探测结果。"""
    section("2. 存储适配层 storage/  —— 真实探测四个外部服务")

    adapters = build_adapters(settings)
    results: list[HealthResult] = []
    try:
        for adapter in adapters:
            result = adapter.health_check()
            results.append(result)
            print("  " + result.render())
    finally:
        for adapter in adapters:
            adapter.close()

    passed = sum(1 for r in results if r.ok)
    print()
    print("  通过 %d / %d" % (passed, len(results)))
    print("  上面每一行都是一次真实的网络调用，没有模拟数据。")

    return results


def show_result_structure(settings: Settings) -> None:
    """打印单个健康检查返回值的内部结构。"""
    section("3. 返回值结构 —— 以 Redis 为例")

    store = RedisStore(settings.redis)
    try:
        result = store.health_check()
    finally:
        store.close()

    print("  service    = %s" % result.service)
    print("  ok         = %s" % result.ok)
    print("  detail     = %s" % result.detail)
    print("  elapsed_ms = %.2f" % result.elapsed_ms)
    print("  render()   = %s" % result.render())
    print()
    print("  服务不可达时 ok=False，detail 里放异常类型与消息，")
    print("  异常不会抛给调用方——这是 health_check 的契约。")


def report_via_logging(results: list[HealthResult]) -> None:
    """用日志输出第 2 段的真实探测结果。

    这里不打任何模拟文本，日志内容就是上面真实调用的结果。
    """
    section("4. 日志 observability/logging.py")

    setup_logging("INFO")
    logger = get_logger("cbe_rag.manual_check")
    print("  级别 INFO：下面每行日志对应一个服务，内容取自真实探测结果")
    print()
    for result in results:
        if result.ok:
            logger.info("%s 连通正常：%s", result.service, result.detail)
        else:
            logger.error("%s 未通过：%s", result.service, result.detail)

    print()
    print("  级别 DEBUG：多输出一条含真实配置值的调试信息")
    print()
    setup_logging("DEBUG")
    first = results[0]
    logger.debug("首个服务的真实探测耗时 %.2f ms", first.elapsed_ms)

    print()
    print("  非法级别名应当报错而不是静默降级：")
    try:
        setup_logging("VERBOSE")
        print("  [FAIL] 没有报错，说明静默降级了")
    except ValueError as exc:
        print("  [OK] 按预期报错：%s" % exc)


def main() -> int:
    print()
    print("已完成模块的真实调用验证")

    settings = show_config()
    if settings is None:
        print()
        print(LINE)
        print("配置未就绪，后续检查无法进行。")
        print(LINE)
        return 1

    results = probe_services(settings)
    show_result_structure(settings)
    report_via_logging(results)

    failed = [r for r in results if not r.ok]
    print()
    print(LINE)
    if failed:
        print("未通过 %d 项：" % len(failed))
        for result in failed:
            print("  - %s：%s" % (result.service, result.detail))
    else:
        print("四个外部服务全部连通。")
    print(LINE)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
