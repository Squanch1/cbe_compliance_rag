"""解析结果质量评估的单元测试。

评估是**决定流程走向的门禁**，不是参考分，因此重点验证：
每个指标都能独立触发失败，且失败信息能定位到具体指标与实测值。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from cbe_rag.ingestion.parser.quality import (
    QualityThresholds,
    assess,
)
from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    ParsedDocument,
    SourceFormat,
)

GOOD_TEXT = "这是一段足够长的正文内容，用来满足字符数下限的要求。" * 12


def paragraph(text: str) -> Block:
    return Block(type=BlockType.PARAGRAPH, text=text, order=0)


def heading(text: str) -> Block:
    return Block(type=BlockType.HEADING, text=text, order=0, level=1)


def document_of(*blocks: Block) -> ParsedDocument:
    return ParsedDocument(
        doc_id="doc-id",
        title="测试文档",
        source_format=SourceFormat.HTML,
        source_path=Path("C:/proj/data/raw/x.html"),
        parser_version="0.1.0",
        parsed_at=datetime(2026, 9, 28),
        blocks=list(blocks) or [paragraph(GOOD_TEXT)],
    )


# 测试用的宽松阈值：只保留要验证的那一条，其余放宽到不可能触发
LOOSE = QualityThresholds(
    min_total_chars=1,
    max_garbage_ratio=1.0,
    max_fragment_ratio=1.0,
    max_duplicate_ratio=1.0,
)


class TestPassing:
    def test_normal_document_passes(self) -> None:
        report = assess(document_of(heading("标题"), paragraph(GOOD_TEXT)))

        assert report.passed is True
        assert report.failures == []

    def test_metrics_are_recorded_even_when_passing(self) -> None:
        # 调阈值要看分布，而分布只能从这些原始分数里来
        report = assess(document_of(paragraph(GOOD_TEXT)))

        assert "total_chars" in report.metrics
        assert "garbage_ratio" in report.metrics
        assert report.metrics["total_chars"] > 0


class TestCharCount:
    def test_too_few_chars_fails(self) -> None:
        limits = QualityThresholds(min_total_chars=10000, max_garbage_ratio=1.0,
                                   max_fragment_ratio=1.0, max_duplicate_ratio=1.0)

        report = assess(document_of(paragraph("短")), limits)

        assert report.passed is False
        assert any("总字符数" in f for f in report.failures)

    def test_failure_names_both_value_and_limit(self) -> None:
        # 只说「不合格」没法定位问题，也没法调阈值
        limits = QualityThresholds(min_total_chars=10000, max_garbage_ratio=1.0,
                                   max_fragment_ratio=1.0, max_duplicate_ratio=1.0)

        report = assess(document_of(paragraph("短")), limits)

        message = report.failures[0]
        assert "10000" in message
        assert "1" in message


class TestGarbageRatio:
    def test_high_garbage_ratio_fails(self) -> None:
        # 字体编码错误的 PDF 会产出大量这类字符
        garbage = "\ufffd\u0001\u0002\u0007" * 50 + GOOD_TEXT
        limits = QualityThresholds(max_garbage_ratio=0.01, min_total_chars=1, max_fragment_ratio=1.0,
                                   max_duplicate_ratio=1.0)

        report = assess(document_of(paragraph(garbage)), limits)

        assert any("乱码率" in f for f in report.failures)

    def test_normal_punctuation_is_not_garbage(self) -> None:
        text = "English, 中文。（括号）【方括号】€150 — dash"
        limits = QualityThresholds(max_garbage_ratio=0.0, min_total_chars=1, max_fragment_ratio=1.0,
                                   max_duplicate_ratio=1.0)

        report = assess(document_of(paragraph(text)), limits)

        assert report.metrics["garbage_ratio"] == 0.0

    def test_empty_text_counts_as_all_garbage(self) -> None:
        # 空文本会在字符数那一项失败，但乱码率也不该悄悄放行
        assert assess(document_of(paragraph("x")), LOOSE).metrics["garbage_ratio"] == 0.0


class TestFragmentRatio:
    def test_many_short_paragraphs_fail(self) -> None:
        # 排版识别失败会把整段拆成一堆碎块
        blocks = [paragraph("碎")] * 10 + [paragraph(GOOD_TEXT)]
        limits = QualityThresholds(max_fragment_ratio=0.3, min_total_chars=1, max_garbage_ratio=1.0,
                                   max_duplicate_ratio=1.0)

        report = assess(document_of(*blocks), limits)

        assert any("碎片块" in f for f in report.failures)

    def test_short_headings_are_not_fragments(self) -> None:
        # 标题本来就短，算作碎片会误判
        blocks = [heading("1.1")] * 10 + [paragraph(GOOD_TEXT)]
        limits = QualityThresholds(max_fragment_ratio=0.0, min_total_chars=1, max_garbage_ratio=1.0,
                                   max_duplicate_ratio=1.0)

        report = assess(document_of(*blocks), limits)

        assert report.metrics["fragment_ratio"] == 0.0


class TestDuplicateRatio:
    def test_repeated_blocks_fail(self) -> None:
        # 页眉页脚没剔干净时，同样的文字会在每页重复出现
        blocks = [paragraph("页眉文字")] * 8 + [paragraph(GOOD_TEXT)]
        limits = QualityThresholds(max_duplicate_ratio=0.3, min_total_chars=1, max_garbage_ratio=1.0,
                                   max_fragment_ratio=1.0)

        report = assess(document_of(*blocks), limits)

        assert any("重复块" in f for f in report.failures)

    def test_distinct_blocks_have_no_duplicates(self) -> None:
        blocks = [paragraph("内容%d" % i) for i in range(10)]

        report = assess(document_of(*blocks), LOOSE)

        assert report.metrics["duplicate_ratio"] == 0.0


class TestMultipleFailures:
    def test_all_failing_metrics_are_reported(self) -> None:
        # 一次报全，避免修一个再跑一次才发现下一个。
        # 构造一份同时踩中四个坑的结果：字符少、片段碎、有乱码、块重复。
        garbage = "�" * 5 + "碎"
        limits = QualityThresholds(
            min_total_chars=10000,
            max_garbage_ratio=0.0,
            max_fragment_ratio=0.0,
            max_duplicate_ratio=0.0,
        )

        report = assess(document_of(*([paragraph(garbage)] * 4)), limits)

        assert len(report.failures) >= 3
