"""初始化存储结构。

用法：
    python scripts/init_storage.py

幂等，可重复执行：
    - 已存在的库、表、集合跳过
    - 维度数据用 ON DUPLICATE KEY UPDATE 覆盖，以 ddl.py 的定义为准

退出码：全部成功 0，有失败 1。

单独做成脚本而不是并进连通性检查，是因为两者的时机不同：
连通性检查每天可能跑，建表只在部署或 schema 变更时跑一次。
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import ENV_FILE_PATH, Settings  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.storage import MilvusStore, MysqlStore  # noqa: E402


def init_mysql(settings: Settings) -> bool:
    """建表并灌入维度数据。返回是否全部成功。"""
    store = MysqlStore(settings.mysql)
    try:
        created = store.create_schema()
        if created:
            print("  [OK] 新建 %d 张表：%s" % (len(created), "、".join(created)))
        else:
            print("  [OK] 表已齐备，无需新建")

        affected = store.seed_dimensions()
        print(
            "  [OK] 维度数据已同步：%s"
            % "、".join("%s %d 行" % (k, v) for k, v in affected.items())
        )
        return True
    except Exception as exc:
        print("  [FAIL] %s: %s" % (type(exc).__name__, str(exc)[:300]))
        return False
    finally:
        store.close()


def init_milvus(settings: Settings) -> bool:
    """建库、建集合。返回是否全部成功。"""
    store = MilvusStore(settings.milvus)
    try:
        if store.ensure_database():
            print("  [OK] 新建数据库：%s" % settings.milvus.database)
        else:
            print("  [OK] 数据库已存在：%s" % settings.milvus.database)

        if store.create_collection(settings.embedding.dense_dim):
            print("  [OK] 新建集合：%s" % settings.milvus.collection)
        else:
            print("  [OK] 集合已存在：%s" % settings.milvus.collection)
        return True
    except Exception as exc:
        print("  [FAIL] %s: %s" % (type(exc).__name__, str(exc)[:300]))
        return False
    finally:
        store.close()


def main() -> int:
    setup_console()

    print("=" * 64)
    print("初始化存储结构")
    print("=" * 64)

    if not ENV_FILE_PATH.is_file():
        print("[FAIL] 找不到配置文件：%s" % ENV_FILE_PATH)
        return 1

    try:
        settings = Settings()
    except ValidationError as exc:
        print("[FAIL] 配置加载失败，缺少或写错了以下配置项：")
        for error in exc.errors():
            print("       - %s" % ".".join(str(p) for p in error["loc"]))
        return 1

    print()
    print("MySQL（%s）" % settings.mysql.database)
    mysql_ok = init_mysql(settings)

    print()
    print("Milvus（%s）" % settings.milvus.database)
    milvus_ok = init_milvus(settings)

    print()
    if mysql_ok and milvus_ok:
        print("存储结构初始化完成。")
        return 0
    print("存在失败项，请先解决再继续。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
