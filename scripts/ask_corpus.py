"""向语料提问，看完整的问答结果。

用法：
    python scripts/ask_corpus.py "进口一站式服务的适用金额上限是多少"
    python scripts/ask_corpus.py --country EU "问题"

不带问题时进入交互模式，逐条输入、回车提问、空行退出。

与 search_corpus.py 的分工：那个只看召回了什么，这个看最终答了什么、
引用了哪里。排查时分清这两件事很省时间——答得不对，可能是检索没召回，
也可能是召回了但生成没用好。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import ENV_FILE_PATH, Settings  # noqa: E402
from cbe_rag.generation import QaResult, answer_question  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.retrieval import RetrievalQuery  # noqa: E402
from cbe_rag.storage import (  # noqa: E402
    BailianClient,
    EmbeddingStore,
    MilvusStore,
    MysqlStore,
)

_LINE = "-" * 60


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


def render_citations(result: QaResult) -> list[str]:
    """把被引用到的材料按文档归并列出来。

    只列真正被引用的，不把召回到的全列上——使用者要核对的是「这句话的
    出处」，不是「检索到了什么」。

    **按文档归并，不按父块列。** 同一份文档的多个父块，标题、出处、生效
    日期都一样，逐条列出来看着像重复；而使用者要核对的恰恰是「出自哪份
    文件」。被引用了几段单独标出来，信息不丢。
    """
    cited = result.answer.citations.cited
    if not cited:
        return []

    grouped: dict[str, list[Any]] = {}
    for parent in cited:
        grouped.setdefault(parent.doc_id, []).append(parent)

    lines = ["引用来源："]
    for index, parents in enumerate(grouped.values(), 1):
        first = parents[0]
        suffix = "（引用 %d 处）" % len(parents) if len(parents) > 1 else ""
        lines.append("  %d. %s%s" % (index, first.title, suffix))
        lines.append("     出处：%s" % (first.source_url or "（未登记）"))
        lines.append(
            "     生效日期：%s"
            % (first.effective_date or "未标注（回答里应已说明）")
        )
    return lines


def render_trace(result: QaResult, elapsed: float) -> list[str]:
    """把这次问答走了哪条路径写清楚。

    拒答与降级是两件事，排查方向也不同：前者要调检索，后者要调提示词或
    模型。只写一句「没答出来」，看的人不知道该往哪边查。
    """
    parents = len(result.retrieval.parents)
    score = result.retrieval.top_score
    return [
        "过程：召回 %d 个父块，稠密最高余弦 %s，用时 %.1f 秒"
        % (parents, "无命中" if score is None else "%.4f" % score, elapsed),
        "路径：%s"
        % (
            "检索质量不足，直接拒答（未调用模型）"
            if result.refused
            else "调用了模型"
            + ("，但回答缺少可核对的出处，已降级" if result.answer.degraded else "")
        ),
    ]


def render_answer(result: QaResult, elapsed: float) -> list[str]:
    """渲染一次问答的完整输出。"""
    lines = render_trace(result, elapsed)
    lines.append("")
    lines.append(result.answer.text)

    citations = render_citations(result)
    if citations:
        lines.append("")
        lines.extend(citations)

    if result.notes:
        lines.append("")
        for note in result.notes:
            lines.append("提示：%s" % note)
    return lines


def run_one(
    text: str,
    args: argparse.Namespace,
    settings: Settings,
    mysql: MysqlStore,
    milvus: MilvusStore,
    embedding: EmbeddingStore,
    client: BailianClient,
) -> None:
    """问一个问题并把完整结果打出来。"""
    query = RetrievalQuery(
        text=text,
        country=args.country,
        doc_type=args.doc_type,
        publisher=args.publisher,
    )

    print("=" * 68)
    print("问题：%s" % query.text)
    print("过滤：%s" % describe_filters(query))
    print("=" * 68)

    started = time.perf_counter()
    result = answer_question(
        query,
        config=settings.retrieval,
        mysql=mysql,
        milvus=milvus,
        embedding=embedding,
        client=client,
    )
    elapsed = time.perf_counter() - started

    print()
    for line in render_answer(result, elapsed):
        print(line)
    print()
    print(_LINE)
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="向语料提问")
    parser.add_argument("question", nargs="?", help="问题；不传则进入交互模式")
    parser.add_argument("--country", help="只查该国家代码，如 EU", default=None)
    parser.add_argument(
        "--doc-type", dest="doc_type", help="只查该文档类型", default=None
    )
    parser.add_argument("--publisher", help="只查该发布机构", default=None)
    return parser.parse_args(argv)


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

    if settings.retrieval.refuse_threshold is None:
        print("[WARN] 拒答阈值未标定，无法判断检索质量，问答会直接报错。")
        print("       先在 .env 里填 CBE_RETRIEVAL__REFUSE_THRESHOLD 再跑。")
        return 1

    mysql = MysqlStore(settings.mysql)
    milvus = MilvusStore(settings.milvus)
    embedding = EmbeddingStore(settings.embedding)
    client = BailianClient(settings.llm)

    try:
        print("加载嵌入模型（首次约 10 秒）...")
        started = time.perf_counter()
        embedding.encode(["预热"])
        print("  完成，用时 %.1f 秒" % (time.perf_counter() - started))
        print()

        if args.question:
            run_one(
                args.question, args, settings, mysql, milvus, embedding, client
            )
            return 0

        print("输入问题后回车提问，直接回车退出。")
        print()
        while True:
            try:
                text = input("问题> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                break
            run_one(text, args, settings, mysql, milvus, embedding, client)
    finally:
        mysql.close()
        milvus.close()
        embedding.close()
        client.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
