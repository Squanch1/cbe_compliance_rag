"""索引流程数据结构的测试。

这些类型本身不含逻辑，值得测的是 `count_by_action` 的统计，
以及「记录不可变」这条约定有没有守住——判重结果被中途改掉
会很难查。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date
from typing import Any

import pytest

from cbe_rag.indexing.models import (
    Decision,
    DocumentOutcome,
    ImportAction,
    ImportReport,
    count_by_action,
)
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

TODAY = date(2026, 9, 29)

_BASE = DocumentRecord(
    doc_id="doc-1",
    content_hash="a" * 64,
    status=DocumentStatus.INDEXED,
    title="欧洲增值税常见问题",
    platform="amazon",
    source_url="https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX",
    publisher="amazon",
    country="EU",
    doc_type="policy",
    effective_date=None,
    collected_date=TODAY,
    raw_path="C:/data/raw/amazon-eu-vat-faq.html",
)


def make_existing(**overrides: Any) -> DocumentRecord:
    """造一条库里的记录，字段可按需覆盖。"""
    return replace(_BASE, **overrides)


def make_outcome(action: ImportAction, **overrides: Any) -> DocumentOutcome:
    """造一份文档的处理结果。"""
    return replace(
        DocumentOutcome(file_name="a.html", action=action, ok=True, detail=""),
        **overrides,
    )


def make_report(outcomes: list[DocumentOutcome]) -> ImportReport:
    """造一份批量报告，只填 outcomes——另两项与统计无关。"""
    return ImportReport(outcomes=outcomes, missing_files=[], orphans=[])


class TestCountByAction:
    def test_counts_only_the_requested_action(self) -> None:
        report = make_report(
            [
                make_outcome(ImportAction.NEW),
                make_outcome(ImportAction.SKIP),
                make_outcome(ImportAction.NEW),
                make_outcome(ImportAction.UPDATE_META),
            ]
        )

        assert count_by_action(report, ImportAction.NEW) == 2
        assert count_by_action(report, ImportAction.SKIP) == 1
        assert count_by_action(report, ImportAction.UPDATE_META) == 1

    def test_action_absent_from_report_counts_zero(self) -> None:
        report = make_report([make_outcome(ImportAction.NEW)])

        assert count_by_action(report, ImportAction.SUPERSEDE_AND_NEW) == 0

    def test_empty_report_counts_zero(self) -> None:
        assert count_by_action(make_report([]), ImportAction.NEW) == 0


class TestDocumentRecord:
    def test_carries_the_doc_id_that_must_be_reused(self) -> None:
        # 判重命中时 doc_id 要取自这里而不是新生成：chunk_id 由 doc_id
        # 派生，换一个就等于在向量库里另起一套前缀
        assert make_existing(doc_id="doc-abc").doc_id == "doc-abc"

    def test_optional_metadata_accepts_none(self) -> None:
        # 元数据不齐的文档也会在库里留下记录，这些字段允许为空
        existing = make_existing(publisher=None, country=None, doc_type=None)

        assert existing.publisher is None
        assert existing.country is None
        assert existing.doc_type is None

    def test_is_immutable(self) -> None:
        existing = make_existing()

        with pytest.raises(FrozenInstanceError):
            existing.doc_id = "other"  # type: ignore[misc]


class TestDecision:
    def test_new_carries_no_records(self) -> None:
        decision = Decision(action=ImportAction.NEW)

        assert decision.matched is None
        assert decision.previous is None

    def test_skip_keeps_the_matched_record(self) -> None:
        # 命中的记录要留着，调用方从它身上取 doc_id
        existing = make_existing()

        decision = Decision(action=ImportAction.SKIP, matched=existing)

        assert decision.matched is existing

    def test_supersede_keeps_the_previous_record(self) -> None:
        # 内容变了，matched 为空（没有任何记录与新内容相同），
        # 要下线的是同 source_url 的旧记录
        previous = make_existing(doc_id="doc-old")

        decision = Decision(
            action=ImportAction.SUPERSEDE_AND_NEW, previous=previous
        )

        assert decision.matched is None
        assert decision.previous is previous


class TestImportAction:
    def test_is_a_string_enum(self) -> None:
        # 值会进日志与报告，用字符串而非数字，读日志时不必回查定义
        assert ImportAction.SUPERSEDE_AND_NEW == "supersede_and_new"

    def test_five_actions_are_defined(self) -> None:
        # 判重只产出这五种动作，多一种意味着 decision.py 漏了分支
        assert len(ImportAction) == 5
