"""按清单把语料导入检索库。

用法：
    python scripts/import_corpus.py

可重复执行：内容与清单登记都没变的文档会被跳过，不会重复入库。

退出码：全部成功 0，有失败 1。

依赖真实服务（MySQL、Milvus）与本地嵌入模型，因此不做成单元测试，
它是运维脚本。纯函数部分（报告渲染）另有测试。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import (  # noqa: E402
    ENV_FILE_PATH,
    PROJECT_ROOT,
    Settings,
)
from cbe_rag.indexing import (  # noqa: E402
    ImportAction,
    ImportReport,
    IndexingContext,
    count_by_action,
    import_documents,
)
from cbe_rag.ingestion.chunker import TokenCounter  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.storage import EmbeddingStore, MilvusStore, MysqlStore  # noqa: E402

# 路径按项目根推算，不依赖当前工作目录（见 CLAUDE.md 5.2）
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def render_report(report: ImportReport) -> list[str]:
    """把导入报告渲染成给人看的文本行。

    分三段：逐份的处理结果、缺失的文件、孤儿清单。

    最后一段最要紧——它需要人来判断，所以要把话说透：这些文档仍在库里
    且可被检索，程序不会自动处理。
    """
    lines: list[str] = []

    lines.append("逐份处理结果（%d 份）" % len(report.outcomes))
    for outcome in report.outcomes:
        lines.append(
            "  [%s] %s  %s"
            % ("OK  " if outcome.ok else "FAIL", outcome.file_name, outcome.detail)
        )

    lines.append("")
    lines.append("动作统计")
    lines.append(
        "  新建 %d、重跑 %d、更新元数据 %d、跳过 %d、版本更替 %d"
        % (
            count_by_action(report, ImportAction.NEW),
            count_by_action(report, ImportAction.REINDEX),
            count_by_action(report, ImportAction.UPDATE_META),
            count_by_action(report, ImportAction.SKIP),
            count_by_action(report, ImportAction.SUPERSEDE_AND_NEW),
        )
    )

    lines.append("")
    lines.append("清单引用了但文件缺失（%d 份）" % len(report.missing_files))
    for item in report.missing_files:
        lines.append(
            "  第 %d 行 %s（找不到 %s）"
            % (item.line_number, item.file_name, item.expected_path)
        )

    lines.append("")
    lines.append("库里有、清单里没有（%d 份）" % len(report.orphans))
    for record in report.orphans:
        lines.append("  %s  %s" % (record.title, record.raw_path))
    if report.orphans:
        lines.append(
            "  这些文档仍在库里，仍会被检索到。要不要下线由你决定，"
            "程序不替你做这个判断。"
        )

    return lines


def main() -> int:
    setup_console()

    print("=" * 64)
    print("导入语料")
    print("=" * 64)

    if not ENV_FILE_PATH.is_file():
        print("[FAIL] 找不到配置文件：%s" % ENV_FILE_PATH)
        return 1
    if not MANIFEST_PATH.is_file():
        print("[FAIL] 找不到清单：%s" % MANIFEST_PATH)
        return 1

    try:
        settings = Settings()
    except ValidationError as exc:
        print("[FAIL] 配置加载失败，缺少或写错了以下配置项：")
        for error in exc.errors():
            print("       - %s" % ".".join(str(p) for p in error["loc"]))
        return 1

    mysql = MysqlStore(settings.mysql)
    milvus = MilvusStore(settings.milvus)
    embedding = EmbeddingStore(settings.embedding)

    try:
        print()
        print("加载分词器与嵌入模型（首次约 10 秒）...")
        started = time.perf_counter()
        counter = TokenCounter(settings.embedding.model_path)
        # 先把模型拉起来，免得它的加载时间混进第一份文档里，看不出来
        embedding.encode(["预热"])
        print("  完成，用时 %.1f 秒" % (time.perf_counter() - started))

        print()
        print("清单：%s" % MANIFEST_PATH)
        print("原始文件：%s" % RAW_DIR)
        print()
        report = import_documents(
            MANIFEST_PATH,
            RAW_DIR,
            IndexingContext(
                mysql=mysql,
                milvus=milvus,
                embedding=embedding,
                counter=counter,
            ),
        )
    finally:
        mysql.close()
        milvus.close()
        embedding.close()

    for line in render_report(report):
        print(line)

    failed = [outcome for outcome in report.outcomes if not outcome.ok]
    print()
    if failed:
        print("有 %d 份没处理成功，见上面的 FAIL 行。" % len(failed))
        return 1
    print("导入完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
