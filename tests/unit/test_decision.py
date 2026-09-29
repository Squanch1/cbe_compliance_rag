"""判重决策的单元测试。

不连数据库：decide 的输入全是现成的事实，因此每条规则都能单独构造
出来验证。判重判错的后果是静默的——重复入库或文档不再被检索到，
两种都不会抛异常，所以这里每种状态组合都要有一条。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

from cbe_rag.indexing.decision import _COMPARED_FIELDS, decide
from cbe_rag.indexing.models import ImportAction
from cbe_rag.ingestion.parser.schema import DocumentMeta
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

TODAY = date(2026, 9, 29)
COLLECTED = date(2026, 9, 20)

_BASE_EXISTING = DocumentRecord(
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
    collected_date=COLLECTED,
    raw_path="C:/data/raw/amazon-eu-vat-faq.html",
)


def existing(**overrides: Any) -> DocumentRecord:
    """造一条库里的记录，字段可按需覆盖。"""
    return replace(_BASE_EXISTING, **overrides)


def meta(**overrides: Any) -> DocumentMeta:
    """造一份清单上的元数据，字段可按需覆盖。

    默认与 _BASE_EXISTING 完全一致，这样「只改一个字段」就能测出
    该字段是否参与了比对。
    """
    fields: dict[str, Any] = {
        "doc_id": "new-doc",
        "title": _BASE_EXISTING.title,
        "collected_date": TODAY,
        "source_url": _BASE_EXISTING.source_url,
        "publisher": _BASE_EXISTING.publisher,
        "country": _BASE_EXISTING.country,
        "doc_type": _BASE_EXISTING.doc_type,
        "effective_date": _BASE_EXISTING.effective_date,
        "platform": _BASE_EXISTING.platform,
    }
    fields.update(overrides)
    return DocumentMeta(**fields)


class TestNew:
    def test_no_match_anywhere_yields_new(self) -> None:
        decision = decide(meta())

        assert decision.action is ImportAction.NEW
        assert decision.matched is None
        assert decision.previous is None


class TestSkip:
    def test_identical_content_and_meta_yields_skip(self) -> None:
        record = existing()

        decision = decide(meta(), by_hash=record)

        assert decision.action is ImportAction.SKIP
        assert decision.matched is record

    def test_collected_date_alone_does_not_trigger_update(self) -> None:
        # 采集日期每次导入都会变，拿它比对等于每次都判为改过，
        # 结果是每次重跑都白更新一遍元数据
        decision = decide(meta(collected_date=TODAY), by_hash=existing())

        assert decision.action is ImportAction.SKIP

    def test_new_doc_id_alone_does_not_trigger_update(self) -> None:
        # doc_id 是每次流水线新生成的，命中的那条要复用它而不是比它
        decision = decide(meta(doc_id="brand-new-id"), by_hash=existing())

        assert decision.action is ImportAction.SKIP


class TestUpdateMeta:
    def test_changed_publisher_yields_update(self) -> None:
        # 最典型的场景：当初留空，后来补上了
        record = existing(publisher=None)

        decision = decide(meta(publisher="eu_commission"), by_hash=record)

        assert decision.action is ImportAction.UPDATE_META
        assert decision.matched is record

    def test_changed_title_yields_update(self) -> None:
        decision = decide(meta(title="改过的标题"), by_hash=existing())

        assert decision.action is ImportAction.UPDATE_META

    def test_filled_source_url_yields_update(self) -> None:
        # 从「没有」变成「有」，也是变化
        record = existing(source_url=None)

        decision = decide(meta(source_url="https://example.org/a"), by_hash=record)

        assert decision.action is ImportAction.UPDATE_META

    def test_cleared_effective_date_yields_update(self) -> None:
        # 从「有」变成「没有」，同样是变化
        record = existing(effective_date=date(2021, 7, 1))

        decision = decide(meta(effective_date=None), by_hash=record)

        assert decision.action is ImportAction.UPDATE_META


class TestReindex:
    def test_pending_yields_reindex(self) -> None:
        # 当初元数据不齐没切分，现在补齐了要重跑
        record = existing(status=DocumentStatus.PENDING)

        decision = decide(meta(), by_hash=record)

        assert decision.action is ImportAction.REINDEX
        assert decision.matched is record

    def test_needs_manual_yields_reindex(self) -> None:
        # 解析器换了一层之后重跑，可能就过了
        record = existing(status=DocumentStatus.NEEDS_MANUAL)

        assert decide(meta(), by_hash=record).action is ImportAction.REINDEX

    def test_failed_yields_reindex(self) -> None:
        record = existing(status=DocumentStatus.FAILED)

        assert decide(meta(), by_hash=record).action is ImportAction.REINDEX

    def test_superseded_yields_reindex(self) -> None:
        # 被新版本取代过的文件又放回了 data/raw。
        # 这里若判成 SKIP，文档会停在不可检索的状态，而报告显示「已跳过」。
        record = existing(status=DocumentStatus.SUPERSEDED)

        assert decide(meta(), by_hash=record).action is ImportAction.REINDEX

    def test_reindex_takes_priority_over_meta_change(self) -> None:
        # 状态没走完时，元数据改了也要重跑——重跑本来就会带上新元数据，
        # 没有理由只更新属性就停下
        record = existing(status=DocumentStatus.PENDING, publisher=None)

        decision = decide(meta(publisher="eu_commission"), by_hash=record)

        assert decision.action is ImportAction.REINDEX


class TestSupersedeAndNew:
    def test_same_url_with_new_content_yields_supersede(self) -> None:
        previous = existing(doc_id="doc-old")

        decision = decide(meta(), by_url=previous)

        assert decision.action is ImportAction.SUPERSEDE_AND_NEW
        assert decision.previous is previous
        # 内容与库里任何一条都不同，因此没有 matched
        assert decision.matched is None

    def test_hash_match_wins_over_url_match(self) -> None:
        # 哈希相同说明文件一个字节都没变，同链接下的另一条记录
        # 不影响本次处理
        hit = existing(doc_id="doc-hit")

        decision = decide(meta(), by_hash=hit, by_url=existing(doc_id="doc-other"))

        assert decision.action is ImportAction.SKIP
        assert decision.matched is hit


class TestComparedFields:
    def test_covers_every_document_meta_field_except_two(self) -> None:
        # 将来给 DocumentMeta 加字段时这条会失败，提醒决定它要不要
        # 参与比对。漏加的后果是改了那个字段而判重看不出来。
        excluded = {"doc_id", "collected_date"}

        assert set(_COMPARED_FIELDS) == set(DocumentMeta.model_fields) - excluded

    def test_excluded_fields_are_still_required_by_document_meta(self) -> None:
        # 排除的两个字段必须真的存在于 DocumentMeta 上，否则上面那条
        # 测试会因为集合运算而悄悄通过
        assert {"doc_id", "collected_date"} <= set(DocumentMeta.model_fields)
