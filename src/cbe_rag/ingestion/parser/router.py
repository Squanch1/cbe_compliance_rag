"""解析路由。

按文件类型与难度组装解析链，逐层尝试并评估质量：
合格就返回，不合格换下一层，全部不合格交人工。

链的长度由**难度**决定，不是每种格式固定几层：
判为简单的文档只挂主力解析器，判为困难的才加上降级层。
"""

from __future__ import annotations

from pathlib import Path

from cbe_rag.ingestion.parser.difficulty import DifficultyReport, probe_difficulty
from cbe_rag.ingestion.parser.quality import QualityThresholds, assess
from cbe_rag.ingestion.parser.schema import SourceFormat
from cbe_rag.ingestion.parser.tier import (
    ParseOutcome,
    ParseRequest,
    TierAttempt,
    TierParser,
)
from cbe_rag.ingestion.parser.tier_html import GenericHtmlTier, SelectorHtmlTier
from cbe_rag.ingestion.parser.tier_pdf import TableAwarePdfTier, TextLayerPdfTier

# 文件头特征。判断类型靠内容不靠扩展名。
_PDF_MAGIC = b"%PDF-"
_HTML_PREFIXES = ("<!doctype", "<html", "<?xml")

# 读取文件头时取的字节数。512 足够覆盖上述特征。
_HEAD_BYTES = 512

# 主力解析器：绝大多数文档靠它一层就能过。
_PRIMARY_TIERS: dict[SourceFormat, TierParser] = {
    SourceFormat.HTML: SelectorHtmlTier(),
    SourceFormat.PDF: TextLayerPdfTier(),
}

# 降级层：只在判为困难时才挂上。
_FALLBACK_TIERS: dict[SourceFormat, tuple[TierParser, ...]] = {
    SourceFormat.HTML: (GenericHtmlTier(),),
    SourceFormat.PDF: (TableAwarePdfTier(),),
}


class ParseRouteError(Exception):
    """路由无法继续时抛出。"""


def detect_format(path: Path) -> SourceFormat:
    """按**文件头**判断类型，扩展名只作参考。

    不能只看扩展名：下载的文件常被浏览器或工具改过名。而且把 PDF
    当 HTML 解析产出的是乱码而不是异常——**猜错比报错难排查得多**，
    所以这里宁可拒绝也不猜。
    """
    if not path.is_absolute():
        raise ParseRouteError("文件路径必须是绝对路径，收到：%s" % path)
    if not path.is_file():
        raise ParseRouteError("文件不存在：%s" % path)

    with path.open("rb") as handle:
        head = handle.read(_HEAD_BYTES)

    if head.startswith(_PDF_MAGIC):
        return SourceFormat.PDF

    text = head.decode("utf-8", "ignore").lstrip("\ufeff \t\r\n").lower()
    if text.startswith(_HTML_PREFIXES) or text.startswith("<"):
        return SourceFormat.HTML

    raise ParseRouteError(
        "无法识别的文件类型（扩展名 %s）。首期只支持 PDF 与 HTML，"
        "请确认文件是否损坏或格式不符" % path.suffix
    )


def build_chain(
    source_format: SourceFormat, difficulty: DifficultyReport
) -> list[TierParser]:
    """按格式与难度组装解析链。

    简单文档只挂主力解析器；困难的才加上降级层。
    """
    chain: list[TierParser] = [_PRIMARY_TIERS[source_format]]
    if difficulty.is_hard:
        chain.extend(_FALLBACK_TIERS[source_format])
    return chain


def parse_document(
    request: ParseRequest,
    *,
    thresholds: QualityThresholds | None = None,
) -> ParseOutcome:
    """解析一份文档：探测难度、组装链、逐层尝试。

    每一层的尝试都记进 attempts——全部失败时，那份记录就是人工处理的
    唯一线索，必须写清机器试过什么、卡在哪。
    """
    source_format = detect_format(request.raw_path)
    difficulty = probe_difficulty(request, source_format)
    chain = build_chain(source_format, difficulty)

    attempts: list[TierAttempt] = []
    for tier in chain:
        try:
            document = tier.parse(request)
        except Exception as exc:
            # 捕获所有异常：失败来源既有解析库的异常，也有「选项里没有
            # 选择器」这类自身的错误。异常类型写进 detail，程序 bug 不会
            # 被藏起来——它会以 AttributeError 之类的形式出现在记录里。
            attempts.append(
                TierAttempt(
                    tier=tier.name,
                    ok=False,
                    detail="%s: %s" % (type(exc).__name__, str(exc)[:200]),
                )
            )
            continue

        report = assess(document, thresholds)
        if report.passed:
            attempts.append(
                TierAttempt(
                    tier=tier.name, ok=True, detail="通过质量评估", report=report
                )
            )
            return ParseOutcome(
                document=document,
                attempts=attempts,
                difficulty=difficulty.level.value,
            )

        attempts.append(
            TierAttempt(
                tier=tier.name,
                ok=False,
                detail="质量不合格：" + "；".join(report.failures),
                report=report,
            )
        )

    return ParseOutcome(
        document=None, attempts=attempts, difficulty=difficulty.level.value
    )
