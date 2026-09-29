"""解析难度探测。

**在解析之前**判断这份文档有多难，据此决定要准备几层解析器：
简单文档一层就够，困难的才挂上降级层。

为什么要在解析前判：解析本身有成本（一份 105 页 PDF 用 PyMuPDF 要
0.3 秒，用 pdfplumber 要 14 秒），而「这份文档难不难」在解析前就能
看出一部分——扫描件的文本层几乎是空的，这不需要跑完解析才知道。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cbe_rag.ingestion.parser.html_parser import HtmlParseError, content_selector_for
from cbe_rag.ingestion.parser.schema import SourceFormat
from cbe_rag.ingestion.parser.tier import ParseRequest

# 抽样页的平均字符数低于此值，判为扫描件。
# 实测有文本层的欧盟 PDF 每页约 2500 字符，扫描件接近 0，阈值取中间很安全。
_SCANNED_PAGE_CHARS = 50

# 抽样页数。取首、中、尾三页——只抽前几页会撞上封面与目录，
# 那几页本来字符就少，会误判成扫描件。
_SAMPLE_PAGES = 3


class DifficultyLevel(str, Enum):
    """解析难度。"""

    SIMPLE = "simple"
    HARD = "hard"


@dataclass(frozen=True)
class DifficultyReport:
    """难度探测结果。

    reasons 逐条记录命中了什么、没命中什么——只给一个「困难」的结论，
    调阈值时无从下手。
    """

    level: DifficultyLevel
    reasons: list[str]

    @property
    def is_hard(self) -> bool:
        return self.level is DifficultyLevel.HARD


def _probe_pdf(path: Path) -> DifficultyReport:
    """探测 PDF 的文本层密度。"""
    import fitz

    reasons: list[str] = []
    try:
        with fitz.open(str(path)) as document:
            total = document.page_count
            if total == 0:
                return DifficultyReport(DifficultyLevel.HARD, ["PDF 没有页面"])

            # 首、中、尾各抽一页。只抽开头会撞上封面与目录而误判。
            indexes = sorted({0, total // 2, total - 1})[:_SAMPLE_PAGES]
            sampled = [len(document[i].get_text().strip()) for i in indexes]

        average = sum(sampled) / len(sampled)
        reasons.append(
            "抽样 %d 页平均 %d 字符（页码 %s）"
            % (len(sampled), int(average), "、".join(str(i + 1) for i in indexes))
        )
        if average < _SCANNED_PAGE_CHARS:
            reasons.append(
                "低于 %d 字符的阈值，判为扫描件——需要 OCR，而首期未实现"
                % _SCANNED_PAGE_CHARS
            )
            return DifficultyReport(DifficultyLevel.HARD, reasons)

        reasons.append("文本层充足，主力解析器应当能处理")
        return DifficultyReport(DifficultyLevel.SIMPLE, reasons)
    except Exception as exc:
        # 打不开的文件本身就是「困难」——交给降级链或人工，
        # 而不是在探测阶段就崩掉
        return DifficultyReport(
            DifficultyLevel.HARD,
            ["无法读取 PDF：%s: %s" % (type(exc).__name__, exc)],
        )


def _probe_html(request: ParseRequest) -> DifficultyReport:
    """探测 HTML 的正文定位难度。

    判据是**站点有没有配置正文容器选择器**，而不是读文件——
    配置表就在内存里，这一探几乎零成本。
    """
    if not request.source_url:
        return DifficultyReport(
            DifficultyLevel.HARD,
            ["缺少 source_url，无法查站点选择器，只能走通用提取"],
        )
    try:
        selector = content_selector_for(request.source_url)
    except HtmlParseError as exc:
        return DifficultyReport(
            DifficultyLevel.HARD, ["站点未配置正文选择器：%s" % str(exc)[:80]]
        )
    return DifficultyReport(
        DifficultyLevel.SIMPLE, ["站点已配置正文容器：%s" % selector]
    )


def probe_difficulty(
    request: ParseRequest,
    source_format: SourceFormat,
) -> DifficultyReport:
    """在解析前判断这份文档有多难。

    两档：简单（主力解析器一层就够）与困难（需要挂上降级层）。
    """
    if source_format is SourceFormat.PDF:
        return _probe_pdf(request.raw_path)
    return _probe_html(request)
