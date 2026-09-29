"""判重准备的单元测试。

MySQL 用假的适配器，不连服务；文件哈希读的是临时目录里的真文件——
「哈希算的到底是什么」正是这一步的关键，换成假哈希就测不出来了。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from cbe_rag.ingestion.fetcher.collector import CollectedDocument
from cbe_rag.ingestion.parser.schema import DocumentMeta
from cbe_rag.indexing.hashing import file_content_hash
from cbe_rag.indexing.models import Decision, ImportAction
from cbe_rag.indexing.pipeline import (
    PreparedDocument,
    prepare_document,
    register_document,
)
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

TODAY = date(2026, 9, 29)
SOURCE_URL = "https://sellercentral.amazon.com/help/hub/reference/GDZ8RCTRUZEH4PBX"


def write_document_file(
    tmp_path: Path,
    content: str = "欧洲增值税常见问题",
    name: str = "amazon-eu-vat-faq.html",
) -> Path:
    """写一个真实的文件，哈希读的就是它。"""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def make_document(raw_path: Path, **overrides: Any) -> CollectedDocument:
    """造一份采集到的文档，元数据字段可按需覆盖。"""
    fields: dict[str, Any] = {
        "doc_id": "new-doc-id",
        "title": "欧洲增值税常见问题",
        "collected_date": TODAY,
        "source_url": SOURCE_URL,
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "platform": "amazon",
    }
    fields.update(overrides)
    return CollectedDocument(raw_path=raw_path, meta=DocumentMeta(**fields))


def make_record(content_hash: str, **overrides: Any) -> DocumentRecord:
    """造一条库里的记录，字段可按需覆盖。"""
    fields: dict[str, Any] = {
        "doc_id": "existing-doc-id",
        "content_hash": content_hash,
        "status": DocumentStatus.INDEXED,
        "title": "欧洲增值税常见问题",
        "platform": "amazon",
        "source_url": SOURCE_URL,
        "publisher": "amazon",
        "country": "EU",
        "doc_type": "faq",
        "effective_date": None,
        "collected_date": TODAY,
        "raw_path": "C:/data/raw/amazon-eu-vat-faq.html",
    }
    fields.update(overrides)
    return DocumentRecord(**fields)


class FakeMysqlStore:
    """假的 MySQL 适配器，只实现判重要用到的两个查询。

    记录查询顺序，用来验证「哈希命中时不该再查 URL」这个取舍。
    """

    def __init__(
        self,
        by_hash: DocumentRecord | None = None,
        by_url: DocumentRecord | None = None,
    ) -> None:
        self._by_hash = by_hash
        self._by_url = by_url
        self.queries: list[str] = []

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        self.queries.append("hash")
        return self._by_hash

    def find_active_by_source_url(self, source_url: str) -> DocumentRecord | None:
        self.queries.append("url")
        return self._by_url


class TestActions:
    def test_unknown_file_is_new(self, tmp_path: Path) -> None:
        document = make_document(write_document_file(tmp_path))

        prepared = prepare_document(document, FakeMysqlStore())

        assert prepared.decision.action is ImportAction.NEW

    def test_same_content_and_meta_is_skip(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)
        record = make_record(file_content_hash(raw))

        prepared = prepare_document(make_document(raw), FakeMysqlStore(by_hash=record))

        assert prepared.decision.action is ImportAction.SKIP

    def test_changed_meta_is_update(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)
        record = make_record(file_content_hash(raw), publisher=None)

        prepared = prepare_document(make_document(raw), FakeMysqlStore(by_hash=record))

        assert prepared.decision.action is ImportAction.UPDATE_META

    def test_unfinished_record_is_reindex(self, tmp_path: Path) -> None:
        # 上次元数据不齐没切分，这次补齐了
        raw = write_document_file(tmp_path)
        record = make_record(file_content_hash(raw), status=DocumentStatus.PENDING)

        prepared = prepare_document(make_document(raw), FakeMysqlStore(by_hash=record))

        assert prepared.decision.action is ImportAction.REINDEX

    def test_same_url_with_new_content_is_supersede(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path, "改过的内容")
        record = make_record("b" * 64)

        prepared = prepare_document(make_document(raw), FakeMysqlStore(by_url=record))

        assert prepared.decision.action is ImportAction.SUPERSEDE_AND_NEW


class TestQueryStrategy:
    def test_hash_hit_does_not_query_by_url(self, tmp_path: Path) -> None:
        # 哈希命中说明文件一个字节都没变，同链接下还有没有别的记录
        # 不影响本次处理，这次查询可以省掉
        raw = write_document_file(tmp_path)
        record = make_record(file_content_hash(raw))
        mysql = FakeMysqlStore(by_hash=record)

        prepare_document(make_document(raw), mysql)

        assert mysql.queries == ["hash"]

    def test_missing_source_url_does_not_query_by_url(self, tmp_path: Path) -> None:
        # source_url 为空时无从查起，也不该拿 None 去查
        raw = write_document_file(tmp_path)
        mysql = FakeMysqlStore()

        prepare_document(make_document(raw, source_url=None), mysql)

        assert mysql.queries == ["hash"]

    def test_hash_miss_queries_by_url(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)
        mysql = FakeMysqlStore()

        prepare_document(make_document(raw), mysql)

        assert mysql.queries == ["hash", "url"]


class TestPreparedFields:
    def test_carries_the_content_hash(self, tmp_path: Path) -> None:
        # 构造 documents 记录时要写它，重算一遍等于把同一个文件读两次
        raw = write_document_file(tmp_path)

        prepared = prepare_document(make_document(raw), FakeMysqlStore())

        assert prepared.content_hash == file_content_hash(raw)

    def test_carries_the_document(self, tmp_path: Path) -> None:
        document = make_document(write_document_file(tmp_path))

        prepared = prepare_document(document, FakeMysqlStore())

        assert prepared.document is document

    def test_hash_follows_file_content(self, tmp_path: Path) -> None:
        # 内容变了哈希就该变，否则判重会把新版本误判成同一份
        first = write_document_file(tmp_path, "内容一", name="a.html")
        second = write_document_file(tmp_path, "内容二", name="b.html")

        hash_first = prepare_document(make_document(first), FakeMysqlStore())
        hash_second = prepare_document(make_document(second), FakeMysqlStore())

        assert hash_first.content_hash != hash_second.content_hash

    def test_renamed_file_keeps_the_same_hash(self, tmp_path: Path) -> None:
        # 判重认内容不认文件名，改了名不该被当成新文档
        first = write_document_file(tmp_path, "同样的内容", name="faq.html")
        second = write_document_file(tmp_path, "同样的内容", name="faq-2026.html")

        prepared_first = prepare_document(make_document(first), FakeMysqlStore())
        prepared_second = prepare_document(make_document(second), FakeMysqlStore())

        assert prepared_first.content_hash == prepared_second.content_hash


class RecordingMysqlStore:
    """记录写操作的假适配器，不连服务。"""

    def __init__(self) -> None:
        self.inserted: list[DocumentRecord] = []
        self.meta_updates: list[DocumentRecord] = []
        self.status_updates: list[tuple[str, DocumentStatus]] = []

    def insert_document(
        self, record: DocumentRecord, *, now: Any = None
    ) -> None:
        self.inserted.append(record)

    def update_document_meta(
        self, record: DocumentRecord, *, now: Any = None
    ) -> None:
        self.meta_updates.append(record)

    def update_document_status(
        self, doc_id: str, status: DocumentStatus, *, now: Any = None
    ) -> None:
        self.status_updates.append((doc_id, status))


def register(
    tmp_path: Path,
    mysql: RecordingMysqlStore,
    *,
    by_hash: DocumentRecord | None = None,
    by_url: DocumentRecord | None = None,
    **meta_overrides: Any,
) -> tuple[DocumentRecord, RecordingMysqlStore]:
    """走一遍「准备 + 登记」，返回登记进去的记录。"""
    raw = write_document_file(tmp_path)
    document = make_document(raw, **meta_overrides)
    prepared = prepare_document(
        document, FakeMysqlStore(by_hash=by_hash, by_url=by_url)
    )
    return register_document(document, prepared, mysql), mysql


class TestRegisterDocument:
    def test_new_document_is_inserted(self, tmp_path: Path) -> None:
        record, mysql = register(tmp_path, RecordingMysqlStore())

        assert mysql.inserted == [record]
        assert mysql.meta_updates == []
        assert mysql.status_updates == []

    def test_new_document_starts_as_pending(self, tmp_path: Path) -> None:
        # 解析结果还不知道，先记 pending；跑成功才改成 indexed
        record, _ = register(tmp_path, RecordingMysqlStore())

        assert record.status is DocumentStatus.PENDING

    def test_new_document_gets_a_fresh_doc_id(self, tmp_path: Path) -> None:
        first, _ = register(tmp_path, RecordingMysqlStore())
        second, _ = register(tmp_path, RecordingMysqlStore())

        assert first.doc_id != second.doc_id
        assert len(first.doc_id) == 36

    def test_reindex_reuses_the_matched_doc_id(self, tmp_path: Path) -> None:
        # 换了 doc_id 就等于让 chunk_id 另起一套前缀，向量库里会出现
        # 同一份文档的两组向量，而旧的那组删不掉
        raw = write_document_file(tmp_path)
        existing = make_record(
            file_content_hash(raw), status=DocumentStatus.PENDING, doc_id="doc-old"
        )

        record, mysql = register(
            tmp_path, RecordingMysqlStore(), by_hash=existing
        )

        assert record.doc_id == "doc-old"
        assert mysql.inserted == []

    def test_reindex_updates_meta_and_status(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)
        existing = make_record(
            file_content_hash(raw), status=DocumentStatus.FAILED, doc_id="doc-old"
        )

        _, mysql = register(tmp_path, RecordingMysqlStore(), by_hash=existing)

        assert len(mysql.meta_updates) == 1
        assert mysql.status_updates == [("doc-old", DocumentStatus.PENDING)]

    def test_reindex_without_matched_record_raises(self, tmp_path: Path) -> None:
        # 判重结果自相矛盾时当场失败，不要写出半条记录
        raw = write_document_file(tmp_path)
        document = make_document(raw)
        prepared = prepare_document(document, FakeMysqlStore())
        broken = PreparedDocument(
            document=document,
            content_hash=prepared.content_hash,
            decision=Decision(action=ImportAction.REINDEX),
        )
        mysql = RecordingMysqlStore()

        with pytest.raises(ValueError, match="REINDEX"):
            register_document(document, broken, mysql)

        assert mysql.inserted == []

    def test_record_carries_the_manifest_metadata(self, tmp_path: Path) -> None:
        record, _ = register(tmp_path, RecordingMysqlStore())

        assert record.title == "欧洲增值税常见问题"
        assert record.source_url == SOURCE_URL
        assert record.publisher == "amazon"
        assert record.country == "EU"
        assert record.doc_type == "faq"
        assert record.platform == "amazon"

    def test_raw_path_is_stored_as_a_string(self, tmp_path: Path) -> None:
        # 表里那列是 VARCHAR，存成 Path 对象的话驱动不知道怎么办
        record, _ = register(tmp_path, RecordingMysqlStore())

        assert isinstance(record.raw_path, str)
        assert record.raw_path.endswith("amazon-eu-vat-faq.html")

    def test_complete_metadata_leaves_missing_fields_empty(self, tmp_path: Path) -> None:
        record, _ = register(tmp_path, RecordingMysqlStore())

        assert record.missing_fields == ()

    def test_incomplete_metadata_is_recorded(self, tmp_path: Path) -> None:
        # 缺哪几项要落到库里，否则手工表里的空缺没人看得见
        record, _ = register(tmp_path, RecordingMysqlStore(), publisher=None)

        assert record.missing_fields == ("publisher",)

    def test_incomplete_metadata_is_still_registered(self, tmp_path: Path) -> None:
        # 门禁在下一步才拦，登记本身照做——这样才查得到「还差哪些」
        record, mysql = register(tmp_path, RecordingMysqlStore(), country=None)

        assert mysql.inserted == [record]

    def test_effective_date_may_be_empty(self, tmp_path: Path) -> None:
        # 很多欧盟指南不标生效日期，它不算必填
        record, _ = register(tmp_path, RecordingMysqlStore(), effective_date=None)

        assert record.effective_date is None
        assert record.missing_fields == ()

    def test_content_hash_is_carried(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)

        record, _ = register(tmp_path, RecordingMysqlStore())

        assert record.content_hash == file_content_hash(raw)
