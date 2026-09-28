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
import time
from pathlib import Path

# 直接以脚本方式运行时，把 src 加入模块搜索路径。
# 通过 pytest 运行时由 pyproject.toml 的 pythonpath 配置提供，此处不生效。
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import Settings  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.storage import MilvusStore, MongoStore, MysqlStore, RedisStore  # noqa: E402
from cbe_rag.storage.health import HealthResult, StorageAdapter  # noqa: E402

# HuggingFace 格式模型的必需文件。
# Ollama 的 GGUF 文件不含这些，因此在进程内加载场景下不可用。
_REQUIRED_MODEL_FILES = ("config.json",)
# 权重文件二选一即可
_WEIGHT_FILE_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def build_adapters(settings: Settings) -> list[StorageAdapter]:
    """按配置构造四个存储适配器。"""
    return [
        MilvusStore(settings.milvus),
        MongoStore(settings.mongodb),
        MysqlStore(settings.mysql),
        RedisStore(settings.redis),
    ]


def check_embedding_model(model_path: Path) -> HealthResult:
    """检查 bge-m3 模型文件是否就位。

    只验证文件结构，不实际加载模型——加载需要数十秒且占用显存，
    不适合放进连通性检查。真正的加载验证在嵌入适配器首次使用时进行。
    """
    started = time.perf_counter()
    model_dir = Path(model_path)

    if not model_dir.is_dir():
        ok = False
        detail = "目录不存在：%s" % model_dir
    else:
        missing = [name for name in _REQUIRED_MODEL_FILES if not (model_dir / name).is_file()]
        has_weight = any((model_dir / name).is_file() for name in _WEIGHT_FILE_CANDIDATES)

        if missing:
            ok = False
            detail = "缺少文件：%s" % "、".join(missing)
        elif not has_weight:
            ok = False
            detail = "缺少权重文件，需包含 %s 之一" % " 或 ".join(_WEIGHT_FILE_CANDIDATES)
        else:
            ok = True
            detail = "路径=%s 精度=fp16" % model_dir

    return HealthResult(
        service="bge-m3",
        ok=ok,
        detail=detail,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


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

    try:
        settings = Settings()
    except ValidationError as exc:
        report_config_error(exc)
        return 1

    adapters = build_adapters(settings)
    try:
        results = [adapter.health_check() for adapter in adapters]
        results.append(check_embedding_model(settings.embedding.model_path))
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
