"""跑评测集，看拒答阈值该定在哪。

用法：
    python scripts/calibrate_threshold.py

对评测集里每道题跑一遍检索，取**稠密路的最高余弦**（拒答判据用的就是它），
按「该答」「该拒答」两类打印分布，并算出能把两类分得最开的候选阈值。

**它只给建议，不替你决定。** 样本少的时候两类的分数区间很可能有重叠，
那时任何阈值都会误伤——报告会把重叠区单独标出来，看到它就该知道这个
阈值靠不住，得先补评测集。

依赖真实服务与本地模型。评测集见 data/eval_cases.csv。
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import ValidationError  # noqa: E402

from cbe_rag.config.settings import ENV_FILE_PATH, Settings  # noqa: E402
from cbe_rag.evaluation import EvalCase, EvalCaseError, load_eval_cases, summarize  # noqa: E402
from cbe_rag.observability import setup_console  # noqa: E402
from cbe_rag.retrieval import RetrievalQuery, search  # noqa: E402
from cbe_rag.storage import EmbeddingStore, MilvusStore, MysqlStore  # noqa: E402


@dataclass(frozen=True)
class CaseScore:
    """一道题的检索结果。"""

    case: EvalCase
    score: float | None
    elapsed_ms: float


@dataclass(frozen=True)
class Suggestion:
    """一条候选粗筛线。"""

    threshold: float
    # 拦住的「该拒答」条数——这才是粗筛想要的
    blocked: int
    # 误伤的「该答」条数——这个必须为 0
    missed: int


def _describe(values: list[float]) -> str:
    """一类分数的分布概况。"""
    if not values:
        return "（没有样本）"
    return "最低 %.4f　中位 %.4f　最高 %.4f" % (
        min(values),
        statistics.median(values),
        max(values),
    )


def suggest_thresholds(
    scores: list[CaseScore], *, max_missed: int = 0
) -> list[Suggestion]:
    """列出所有不误伤的粗筛线，按拦住的条数从多到少排。

    **返回的是全部可选线，不只最优那条。** 拦得最多的那条往往贴着「该答的
    最低分」，没有一点余量——多一道分稍低的该答题就会误伤。只给一条会让
    人以为它稳当，所以把各级线都摆出来，由人按愿意留多少余量来选。

    **约束是不误伤**：误伤的「该答」条数不能超过 max_missed（默认 0，一条
    都不许）。这两件事的代价不对称——漏拦只是多调一次模型，误伤是用户拿不到
    本来能答的问题。

    候选取各条的分数：把线放在某条的分数上，那条算「够」。

    **没有命中（score 为 None）按最低分处理**，因此它一定会被拦下。
    该答的遇到这种情况算误伤，会被 max_missed 排除掉。
    """
    if not scores:
        return []

    def value(item: CaseScore) -> float:
        return item.score if item.score is not None else -1.0

    suggestions: list[Suggestion] = []
    for threshold in sorted({value(item) for item in scores}, reverse=True):
        blocked = 0
        missed = 0
        for item in scores:
            if value(item) >= threshold:
                continue
            if item.case.should_answer:
                missed += 1
            else:
                blocked += 1
        if missed <= max_missed:
            suggestions.append(
                Suggestion(threshold=threshold, blocked=blocked, missed=missed)
            )

    return sorted(suggestions, key=lambda item: -item.blocked)


def render(scores: list[CaseScore], suggestions: list[Suggestion]) -> list[str]:
    """把标定结果渲染成给人看的几行。"""
    summary = summarize([item.case for item in scores])
    should_answer = [item for item in scores if item.case.should_answer]
    should_refuse = [item for item in scores if not item.case.should_answer]

    answer_values = [item.score for item in should_answer if item.score is not None]
    refuse_values = [item.score for item in should_refuse if item.score is not None]

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("评测集：共 %d 道，该答 %d、该拒答 %d" % (
        summary.total, summary.answer_count, summary.refuse_count))
    lines.append("=" * 72)
    lines.append("")

    lines.append("该答的（分数应当高）：")
    for item in sorted(should_answer, key=lambda x: -(x.score or -1)):
        lines.append("  %s  %s" % (
            "无命中" if item.score is None else "%.4f" % item.score,
            item.case.question,
        ))
    lines.append("")
    lines.append("  分布：%s" % _describe(answer_values))
    lines.append("")

    lines.append("该拒答的（分数应当低）：")
    for item in sorted(should_refuse, key=lambda x: -(x.score or -1)):
        lines.append("  %s  %s" % (
            "无命中" if item.score is None else "%.4f" % item.score,
            item.case.question,
        ))
    lines.append("")
    lines.append("  分布：%s" % _describe(refuse_values))
    lines.append("")

    # 能不能分开，看的是「该答的分数是不是全都高于该拒答的」。
    #
    # **不能只看区间有没有数值重叠**：该答 [0.50]、该拒 [0.55] 两个区间一点
    # 不交叠，但顺序反了，照样没有阈值分得开。
    if answer_values and refuse_values:
        worst_answer = max(refuse_values)  # 该拒答里最高的那条
        best_refuse = min(answer_values)   # 该答里最低的那条
        if best_refuse > worst_answer:
            lines.append(
                "该答的分数全部高于该拒答的——阈值取 %.4f 到 %.4f 之间都能分开。"
                % (worst_answer, best_refuse)
            )
        else:
            low_answer = [v for v in answer_values if v <= worst_answer]
            high_refuse = [v for v in refuse_values if v >= best_refuse]
            lines.append(
                "两类分不开：该答里有 %d 条不高于该拒答的最高分（%.4f），"
                "该拒答里有 %d 条不低于该答的最低分（%.4f）。"
                % (len(low_answer), worst_answer, len(high_refuse), best_refuse)
            )
            lines.append("  这些题无论阈值定在哪儿都会被判错。")
        lines.append("")

    if not suggestions:
        lines.append("评测集是空的，没法定阈值。")
        return lines

    lowest_answer = min(answer_values) if answer_values else None
    lines.append("可选的粗筛线（都不误伤）：")
    lines.append("")
    for item in suggestions:
        note = ""
        if lowest_answer is not None and item.threshold == lowest_answer:
            # 贴边的那条最危险：再多一道分稍低的该答题就会误伤
            note = "　← 贴着该答的最低分，没有余量"
        lines.append(
            "  线 %.4f　拦住 %d 条　余量 %.4f%s"
            % (
                item.threshold,
                item.blocked,
                (lowest_answer - item.threshold)
                if lowest_answer is not None
                else 0.0,
                note,
            )
        )
    lines.append("")
    lines.append("余量 = 这条线离「该答的最低分」还有多远。余量越小，")
    lines.append("越容易被一道分稍低的该答题目撞上——那题本来能答，却被拦了。")
    lines.append("")
    lines.append("这条线只拦明显无关的问题，省下一次模型调用；")
    lines.append("真正判断「材料够不够回答」交给模型——实测它比稠密向量靠谱。")
    lines.append("")
    lines.append("注意：要让两类完全分得开是做不到的。稠密相似度分不出")
    lines.append("「主题相近但没答案」和「有答案」——问关税比问增值税阈值")
    lines.append("分数还高，而前者语料里根本没有答案。")
    return lines


def main() -> int:
    setup_console()

    if not ENV_FILE_PATH.is_file():
        print("[FAIL] 找不到配置文件：%s" % ENV_FILE_PATH)
        return 1

    try:
        cases = load_eval_cases()
    except EvalCaseError as exc:
        print("[FAIL] %s" % exc)
        return 1

    if not cases:
        print("[FAIL] 评测集是空的：data/eval_cases.csv")
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

        scores: list[CaseScore] = []
        for case in cases:
            started = time.perf_counter()
            outcome = search(
                RetrievalQuery(text=case.question),
                milvus=milvus,
                embedding=embedding,
                config=settings.retrieval,
            )
            scores.append(
                CaseScore(
                    case=case,
                    score=outcome.dense_top_score,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
    finally:
        mysql.close()
        milvus.close()
        embedding.close()

    for line in render(scores, suggest_thresholds(scores)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
