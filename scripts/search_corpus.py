"""在语料里检索，把每一步的结果打出来。

用法：
    python scripts/search_corpus.py "进口一站式服务的适用金额上限是多少"
    python scripts/search_corpus.py --country EU --doc-type guideline "问题"

不带问题时进入交互模式，逐条输入、回车检索、空行退出。

这是给人看的脚本，输出以看得清为准：每步的耗时、召回的是哪几段、
分数多少、正文讲什么，都打出来。它依赖真实服务与本地模型。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import ENV_FILE_PATH, RetrievalConfig, Settings  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.retrieval import (  # noqa: E402
    RetrievalQuery,
    RetrievedParent,
    load_parents,
    search,
)
from cbe_rag.storage import EmbeddingStore, MilvusStore, MysqlStore  # noqa: E402

# 正文预览的长度。够看清这段在讲什么就行，不必把父块全文铺满屏幕。
PREVIEW_CHARS = 240

_LINE = "-" * 60


def preview(text: str, *, limit: int = PREVIEW_CHARS) -> list[str]:
    """把父块正文压成几行，保留可读的片段。

    换行与连续空白先压成单个空格：父块正文里带着排版留下的换行，
    原样打出来会碎成一堆短行。
    """
    flat = " ".join(text.split())
    if not flat:
        return ["（空）"]
    if len(flat) <= limit:
        return [flat]
    return [flat[:limit] + "……", "（共 %d 字符）" % len(flat)]


def render_parents(parents: list[RetrievedParent]) -> list[str]:
    """把恢复出来的父块渲染成给人看的几行。"""
    lines: list[str] = []
    for index, parent in enumerate(parents, 1):
        lines.append("  %d.  %.4f  %s" % (index, parent.score, parent.title))
        lines.append("      出处：%s" % (parent.source_url or "（未登记）"))
        lines.append(
            "      生效日期：%s"
            % (parent.effective_date or "未标注（回答时必须写明）")
        )
        lines.append(
            "      维度：%s / %s / %s"
            % (parent.country, parent.doc_type, parent.publisher)
        )
        # token 数标出来：它才是喂给模型的那个量，字符数只是便于看正文
        lines.append("      规模：%d token" % parent.token_count)
        lines.append("      %s" % _LINE)
        lines.extend("      %s" % line for line in preview(parent.text))
        lines.append("")
    return lines


def render_verdict(top_score: float | None, config: RetrievalConfig) -> list[str]:
    """渲染质量判据与拒答判定。

    阈值未标定时**不猜**，直接把「判不了」说出来——这正是
    passes_prefilter 会抛错的理由。
    """
    if top_score is None:
        lines = ["质量判据（稠密路最高余弦）：没有任何命中"]
    else:
        lines = ["质量判据（稠密路最高余弦）：%.4f" % top_score]

    if config.refuse_threshold is None:
        lines.append(
            "  拒答阈值未标定，判不了。这是刻意的：未经校准的检索结果"
            "不该直接放行。"
        )
        return lines

    lines.append("  拒答阈值：%.4f" % config.refuse_threshold)
    sufficient = top_score is not None and top_score >= config.refuse_threshold
    lines.append("  判定：%s" % ("证据够" if sufficient else "证据不够，应走拒答"))
    return lines


def _budget_note(total_tokens: int) -> str:
    """给一个「这些上下文占多大地方」的粗略参照。

    千问系列的上下文窗口按 32k 起，这里只给个数量感——具体能装多少由
    生成层按模型实际窗口决定。
    """
    if total_tokens < 8000:
        return "不到四分之一"
    if total_tokens < 16000:
        return "约一半"
    return "接近满"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="在语料里检索")
    parser.add_argument("question", nargs="?", help="问题；不传则进入交互模式")
    parser.add_argument("--country", help="只查该国家代码，如 EU", default=None)
    parser.add_argument("--doc-type", dest="doc_type", help="只查该文档类型", default=None)
    parser.add_argument("--publisher", help="只查该发布机构", default=None)
    return parser.parse_args(argv)


def build_query(text: str, args: argparse.Namespace) -> RetrievalQuery:
    """按命令行参数组装检索请求。"""
    return RetrievalQuery(
        text=text,
        country=args.country,
        doc_type=args.doc_type,
        publisher=args.publisher,
    )


def describe_filters(query: RetrievalQuery) -> str:
    """把过滤条件写成一行，便于确认到底筛了什么。"""
    parts = [
        "%s=%s" % (name, value)
        for name, value in (
            ("国家", query.country),
            ("类型", query.doc_type),
            ("机构", query.publisher),
        )
        if value is not None
    ]
    return "、".join(parts) if parts else "无"


def run_one(
    text: str,
    args: argparse.Namespace,
    settings: Settings,
    mysql: MysqlStore,
    milvus: MilvusStore,
    embedding: EmbeddingStore,
) -> None:
    """检索一个问题并把每一步的结果打出来。"""
    query = build_query(text, args)

    print("=" * 64)
    print("问题：%s" % query.text)
    print("过滤：%s" % describe_filters(query))
    print("=" * 64)
    print()

    started = time.perf_counter()
    outcome = search(
        query, milvus=milvus, embedding=embedding, config=settings.retrieval
    )
    print(
        "[1/2] 混合检索（稠密 %d / 稀疏 %d，权重 %.1f/%.1f）"
        % (
            settings.retrieval.dense_limit,
            settings.retrieval.sparse_limit,
            settings.retrieval.dense_weight,
            settings.retrieval.sparse_weight,
        )
    )
    print(
        "      折叠后 %d 个父块，用时 %.2f 秒"
        % (len(outcome.parents), time.perf_counter() - started)
    )
    print()

    started = time.perf_counter()
    parents = load_parents(outcome.parents, mysql)
    print("[2/2] 按 parent_id 恢复父块")
    print("      用时 %.2f 秒" % (time.perf_counter() - started))
    print()

    if not parents:
        print("  没有任何命中。")
        print()
    else:
        total = sum(parent.token_count for parent in parents)
        print("召回结果（按分数从高到低）")
        print("  合计 %d token，约占提示词预算的 %s" % (total, _budget_note(total)))
        print()
        for line in render_parents(parents):
            print(line)

    for line in render_verdict(outcome.dense_top_score, settings.retrieval):
        print(line)
    print()


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = parse_args(argv)

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

    mysql = MysqlStore(settings.mysql)
    milvus = MilvusStore(settings.milvus)
    embedding = EmbeddingStore(settings.embedding)

    try:
        print("加载嵌入模型（首次约 10 秒）...")
        started = time.perf_counter()
        embedding.encode(["预热"])
        print("  完成，用时 %.1f 秒" % (time.perf_counter() - started))
        print()

        if args.question:
            run_one(args.question, args, settings, mysql, milvus, embedding)
            return 0

        print("输入问题后回车检索，直接回车退出。")
        print()
        while True:
            try:
                text = input("问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                break
            run_one(text, args, settings, mysql, milvus, embedding)
    finally:
        mysql.close()
        milvus.close()
        embedding.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
