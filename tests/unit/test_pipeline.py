"""索引流程的单元测试。

MySQL、Milvus 与嵌入都用假实现，不连服务；文件哈希读的是临时目录里的
真文件——「哈希算的到底是什么」正是判重的关键，换成假哈希就测不出来了。

判错不会报错，只会让同一份文档重复入库，或者让它悄悄不再被检索到，
所以每条分支都要有一条测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cbe_rag.indexing.hashing import file_content_hash
from cbe_rag.indexing.models import Decision, ImportAction
from cbe_rag.indexing.pipeline import (
    PreparedDocument,
    build_vectors,
    index_document,
    prepare_document,
    register_document,
)
from cbe_rag.storage.ddl import DocumentStatus
from indexing_fakes import (
    HTML_WITH_CONTENT,
    SOURCE_URL,
    FakeEmbeddingStore,
    FakeMysqlStore,
    RecordingMilvusStore,
    RecordingMysqlStore,
    analyze,
    make_child,
    make_context,
    make_document,
    make_record,
    register,
    write_document_file,
    write_html_file,
)


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


class TestRegisterDocument:
    def test_new_document_is_inserted(self, tmp_path: Path) -> None:
        mysql = RecordingMysqlStore()

        record = register(tmp_path, mysql)

        assert mysql.inserted == [record]
        assert mysql.meta_updates == []
        assert mysql.status_updates == []

    def test_new_document_starts_as_pending(self, tmp_path: Path) -> None:
        # 解析结果还不知道，先记 pending；跑成功才改成 indexed
        record = register(tmp_path, RecordingMysqlStore())

        assert record.status is DocumentStatus.PENDING

    def test_new_document_gets_a_fresh_doc_id(self, tmp_path: Path) -> None:
        first = register(tmp_path, RecordingMysqlStore())
        second = register(tmp_path, RecordingMysqlStore())

        assert first.doc_id != second.doc_id
        assert len(first.doc_id) == 36

    def test_reindex_reuses_the_matched_doc_id(self, tmp_path: Path) -> None:
        # 换了 doc_id 就等于让 chunk_id 另起一套前缀，向量库里会出现
        # 同一份文档的两组向量，而旧的那组删不掉
        raw = write_document_file(tmp_path)
        existing = make_record(
            file_content_hash(raw), status=DocumentStatus.PENDING, doc_id="doc-old"
        )
        mysql = RecordingMysqlStore()

        record = register(tmp_path, mysql, by_hash=existing)

        assert record.doc_id == "doc-old"
        assert mysql.inserted == []

    def test_reindex_updates_meta_and_status(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)
        existing = make_record(
            file_content_hash(raw), status=DocumentStatus.FAILED, doc_id="doc-old"
        )
        mysql = RecordingMysqlStore()

        register(tmp_path, mysql, by_hash=existing)

        assert len(mysql.meta_updates) == 1
        assert mysql.status_updates == [("doc-old", DocumentStatus.PENDING)]

    def test_reindex_without_matched_record_raises(self, tmp_path: Path) -> None:
        # 判重结果自相矛盾时当场失败，不要写出半条记录
        document = make_document(write_document_file(tmp_path))
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
        record = register(tmp_path, RecordingMysqlStore())

        assert record.title == "欧洲增值税常见问题"
        assert record.source_url == SOURCE_URL
        assert record.publisher == "amazon"
        assert record.country == "EU"
        assert record.doc_type == "faq"
        assert record.platform == "amazon"

    def test_raw_path_is_stored_as_a_string(self, tmp_path: Path) -> None:
        # 表里那列是 VARCHAR，存成 Path 对象的话驱动不知道怎么办
        record = register(tmp_path, RecordingMysqlStore())

        assert isinstance(record.raw_path, str)
        assert record.raw_path.endswith("amazon-eu-vat-faq.html")

    def test_complete_metadata_leaves_missing_fields_empty(self, tmp_path: Path) -> None:
        record = register(tmp_path, RecordingMysqlStore())

        assert record.missing_fields == ()

    def test_incomplete_metadata_is_recorded(self, tmp_path: Path) -> None:
        # 缺哪几项要落到库里，否则手工表里的空缺没人看得见
        record = register(tmp_path, RecordingMysqlStore(), publisher=None)

        assert record.missing_fields == ("publisher",)

    def test_incomplete_metadata_is_still_registered(self, tmp_path: Path) -> None:
        # 门禁在下一步才拦，登记本身照做——这样才查得到「还差哪些」
        mysql = RecordingMysqlStore()

        record = register(tmp_path, mysql, country=None)

        assert mysql.inserted == [record]

    def test_effective_date_may_be_empty(self, tmp_path: Path) -> None:
        # 很多欧盟指南不标生效日期，它不算必填
        record = register(tmp_path, RecordingMysqlStore(), effective_date=None)

        assert record.effective_date is None
        assert record.missing_fields == ()

    def test_content_hash_is_carried(self, tmp_path: Path) -> None:
        raw = write_document_file(tmp_path)

        record = register(tmp_path, RecordingMysqlStore())

        assert record.content_hash == file_content_hash(raw)


class TestBuildVectors:
    def test_encodes_the_child_texts(self) -> None:
        embedding = FakeEmbeddingStore()
        children = [make_child(text="第一段"), make_child(text="第二段", chunk_id="b")]

        build_vectors(children, make_record("a" * 64), embedding)

        assert embedding.encoded == [["第一段", "第二段"]]

    def test_returns_one_vector_per_child(self) -> None:
        embedding = FakeEmbeddingStore()
        children = [
            make_child(chunk_id="a"),
            make_child(chunk_id="b"),
            make_child(chunk_id="c"),
        ]

        vectors = build_vectors(children, make_record("a" * 64), embedding)

        assert [vector.chunk_id for vector in vectors] == ["a", "b", "c"]

    def test_vectors_are_not_shifted(self) -> None:
        # 第 n 个子块应配第 n 条向量。错位不报错，只是检索结果整体偏掉。
        embedding = FakeEmbeddingStore()
        children = [make_child(chunk_id="a"), make_child(chunk_id="b")]

        vectors = build_vectors(children, make_record("a" * 64), embedding)

        assert vectors[0].dense == [0.0, 0.0, 0.0]
        assert vectors[1].dense == [1.0, 1.0, 1.0]
        assert vectors[0].sparse == {0: 1.0}
        assert vectors[1].sparse == {1: 1.0}

    def test_carries_the_parent_id(self) -> None:
        # 检索召回子块后靠它回 MySQL 取父块全文
        embedding = FakeEmbeddingStore()
        children = [make_child(parent_id="doc-1_p0007")]

        vectors = build_vectors(children, make_record("a" * 64), embedding)

        assert vectors[0].parent_id == "doc-1_p0007"

    def test_fills_the_filter_dimensions_from_the_record(self) -> None:
        # 三个维度冗余进 Milvus，检索才不用先回 MySQL 查一遍
        embedding = FakeEmbeddingStore()
        record = make_record(
            "a" * 64, country="DE", doc_type="policy", publisher="eu_commission"
        )

        vectors = build_vectors([make_child()], record, embedding)

        assert vectors[0].country == "DE"
        assert vectors[0].doc_type == "policy"
        assert vectors[0].publisher == "eu_commission"

    def test_empty_children_sends_nothing(self) -> None:
        embedding = FakeEmbeddingStore()

        assert build_vectors([], make_record("a" * 64), embedding) == []
        assert embedding.encoded == []

    def test_vector_count_mismatch_raises(self) -> None:
        # 少了向量就该报错：zip 会静默丢掉末尾的子块，
        # 那些内容在 MySQL 里有、向量库里没有，永远检索不到
        embedding = FakeEmbeddingStore(vector_count=1)
        children = [make_child(chunk_id="a"), make_child(chunk_id="b")]

        with pytest.raises(ValueError, match="不符"):
            build_vectors(children, make_record("a" * 64), embedding)

    def test_extra_vectors_raise_too(self) -> None:
        embedding = FakeEmbeddingStore(vector_count=3)
        children = [make_child(chunk_id="a")]

        with pytest.raises(ValueError, match="不符"):
            build_vectors(children, make_record("a" * 64), embedding)


class TestMetadataGate:
    def test_incomplete_metadata_is_not_indexed(self, tmp_path: Path) -> None:
        # 切片进了向量库就会被检索到并挂进引用里
        context = make_context()

        _, record = analyze(tmp_path, context, publisher=None)

        assert record.missing_fields == ("publisher",)
        assert context.milvus.upserted == []
        assert context.mysql.chunks_written == []

    def test_incomplete_metadata_counts_as_handled(self, tmp_path: Path) -> None:
        # 「没入库」不算失败——文档确实收下了，只是还差信息等人补
        context = make_context()

        result, _ = analyze(tmp_path, context, publisher=None)

        assert result.ok is True
        assert "publisher" in result.detail

    def test_incomplete_metadata_leaves_the_status_alone(self, tmp_path: Path) -> None:
        # 停在 pending 等人补齐，不该被拨成别的状态
        context = make_context()

        analyze(tmp_path, context, publisher=None)

        assert context.mysql.status_updates == []


class TestSuccessfulIndex:
    def test_writes_both_levels_to_mysql(self, tmp_path: Path) -> None:
        context = make_context()

        result, record = analyze(tmp_path, context)

        assert len(context.mysql.chunks_written) == 1
        doc_id, chunks = context.mysql.chunks_written[0]
        assert doc_id == record.doc_id
        assert len(chunks) == result.parent_count + result.child_count

    def test_writes_only_children_to_milvus(self, tmp_path: Path) -> None:
        # 父块不建向量，靠 parent_id 回 MySQL 取全文
        context = make_context()

        result, _ = analyze(tmp_path, context)

        assert len(context.milvus.upserted[0]) == result.child_count

    def test_clears_previous_vectors_before_writing(self, tmp_path: Path) -> None:
        # 分块数量可能变少，光靠 upsert 清不掉多出来的那几条
        context = make_context()

        _, record = analyze(tmp_path, context)

        assert context.milvus.deleted == [record.doc_id]

    def test_marks_the_document_indexed(self, tmp_path: Path) -> None:
        context = make_context()

        _, record = analyze(tmp_path, context)

        assert (record.doc_id, DocumentStatus.INDEXED) in context.mysql.status_updates

    def test_reports_ok_with_block_counts(self, tmp_path: Path) -> None:
        context = make_context()

        result, _ = analyze(tmp_path, context)

        assert result.ok is True
        assert result.parent_count >= 1
        assert result.child_count >= 1

    def test_child_vectors_carry_the_filter_dimensions(self, tmp_path: Path) -> None:
        context = make_context()

        analyze(tmp_path, context)
        vector = context.milvus.upserted[0][0]

        assert vector.country == "EU"
        assert vector.doc_type == "faq"
        assert vector.publisher == "amazon"

    def test_child_vectors_point_back_at_a_parent_that_exists(
        self, tmp_path: Path
    ) -> None:
        # 检索召回子块后靠 parent_id 回 MySQL 取父块全文，
        # 指向一个不存在的父块就等于拿不到上下文
        context = make_context()

        analyze(tmp_path, context)
        _, chunks = context.mysql.chunks_written[0]
        parent_ids = {
            chunk.chunk_id
            for chunk in chunks
            if chunk.level.value == "parent"
        }

        assert parent_ids
        assert all(
            vector.parent_id in parent_ids for vector in context.milvus.upserted[0]
        )


class TestSiblingRetirement:
    def test_retires_siblings_and_deletes_their_vectors(self, tmp_path: Path) -> None:
        # 光改状态拦不住检索：Milvus 里没有 status 字段，
        # 旧记录的向量照样会被召回
        context = make_context(mysql=RecordingMysqlStore(siblings=["doc-old"]))

        _, record = analyze(tmp_path, context)

        assert context.mysql.sibling_lookups == [(SOURCE_URL, record.doc_id)]
        assert context.milvus.deleted == [record.doc_id, "doc-old"]

    def test_no_siblings_means_no_extra_deletes(self, tmp_path: Path) -> None:
        context = make_context()

        _, record = analyze(tmp_path, context)

        assert context.milvus.deleted == [record.doc_id]


# 正文不在 help-content 容器里，选择器取不到内容。
# 这一层是抛异常失败的，还没走到质量评估，因此没有 report。
UNPARSABLE = (
    "<!doctype html><html><body>"
    "<div id='other'>正文在别的容器里，选择器取不到</div>"
    "</body></html>"
)

# 能解析出内容，但总字符数远低于质量门禁的下限（200）。
# 这一层走到了质量评估才失败，因此带 report。
#
# 必须有 <h1> 或 <title>：解析器在取标题那一步就会报错，没有标题的话
# 根本走不到质量评估，测的就不是这条路径了。
TOO_SHORT = (
    "<!doctype html><html><body><div id='help-content'>"
    "<h1>短文档</h1><p>太短</p></div></body></html>"
)


class TestParseFailure:

    def test_unparsable_content_marks_needs_manual(self, tmp_path: Path) -> None:
        context = make_context()

        result, record = analyze(tmp_path, context, body=UNPARSABLE)

        assert result.ok is False
        assert (
            record.doc_id,
            DocumentStatus.NEEDS_MANUAL,
        ) in context.mysql.status_updates

    def test_nothing_is_written_to_storage(self, tmp_path: Path) -> None:
        context = make_context()

        analyze(tmp_path, context, body=UNPARSABLE)

        assert context.milvus.upserted == []
        assert context.mysql.chunks_written == []

    def test_message_names_the_tiers_that_were_tried(self, tmp_path: Path) -> None:
        # 人工处理时最想知道机器试过什么、卡在哪
        context = make_context()

        result, _ = analyze(tmp_path, context, body=UNPARSABLE)

        assert "html.selector" in result.detail
        assert "交人工" in result.detail


class TestParseAttemptsAreRecorded:
    def test_failure_records_every_tier_that_ran(self, tmp_path: Path) -> None:
        context = make_context()

        analyze(tmp_path, context, body=UNPARSABLE)

        written = context.mysql.parse_attempts_written[0][1]
        assert written
        assert written[0]["tier"] == "html.selector"

    def test_failure_records_why_it_failed(self, tmp_path: Path) -> None:
        context = make_context()

        analyze(tmp_path, context, body=UNPARSABLE)

        written = context.mysql.parse_attempts_written[0][1]
        assert written[0]["ok"] is False
        assert written[0]["detail"]

    def test_quality_failure_records_the_metrics(self, tmp_path: Path) -> None:
        # 调阈值时要看分布，而分布只能从这些原始分数里来
        context = make_context()

        analyze(tmp_path, context, body=TOO_SHORT)

        written = context.mysql.parse_attempts_written[0][1]
        assert written[0]["metrics"]
        assert written[0]["failures"]

    def test_exception_failure_has_no_metrics(self, tmp_path: Path) -> None:
        # 抛异常的那层还没走到质量评估，没有分数可记。
        # 这里不是缺陷：detail 里写着异常类型，那才是排查线索。
        context = make_context()

        analyze(tmp_path, context, body=UNPARSABLE)

        written = context.mysql.parse_attempts_written[0][1]
        assert written[0]["metrics"] == {}
        assert written[0]["detail"]

    def test_success_also_records_the_attempts(self, tmp_path: Path) -> None:
        # 「这份文档是哪一层解析出来的」对排查别的质量问题同样有用
        context = make_context()

        analyze(tmp_path, context)

        written = context.mysql.parse_attempts_written[0][1]
        assert written[0]["ok"] is True

    def test_metadata_gate_writes_no_attempts(self, tmp_path: Path) -> None:
        # 没进解析就没有尝试记录，那一列该是空的
        context = make_context()

        analyze(tmp_path, context, publisher=None)

        assert context.mysql.parse_attempts_written == []


class TestSkipThroughIndexDocument:
    def test_unchanged_document_writes_nothing(self, tmp_path: Path) -> None:
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), doc_id="doc-1")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        outcome = index_document(make_document(raw), context)

        assert outcome.action is ImportAction.SKIP
        assert outcome.ok is True
        assert outcome.doc_id == "doc-1"
        assert context.mysql.chunks_written == []
        assert context.milvus.upserted == []
        assert context.milvus.deleted == []

    def test_skip_reuses_the_existing_doc_id(self, tmp_path: Path) -> None:
        # 报告里的 doc_id 要能直接对回库里那条，不然没法查
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), doc_id="doc-existing")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        assert index_document(make_document(raw), context).doc_id == "doc-existing"


class TestUpdateMetaThroughIndexDocument:
    def test_updates_mysql_metadata(self, tmp_path: Path) -> None:
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), publisher=None, doc_id="doc-1")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        outcome = index_document(
            make_document(raw, publisher="eu_commission"), context
        )

        assert outcome.action is ImportAction.UPDATE_META
        assert context.mysql.meta_updates[0].publisher == "eu_commission"

    def test_syncs_the_filter_dimensions_in_milvus(self, tmp_path: Path) -> None:
        # 三个维度在 Milvus 里冗余存了一份，不同步的话按 publisher 筛
        # 会漏掉这份文档——它的向量还在用旧值
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), publisher=None, doc_id="doc-1")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        index_document(make_document(raw, publisher="eu_commission"), context)

        assert context.milvus.scalar_updates == [
            {
                "doc_id": "doc-1",
                "country": "EU",
                "doc_type": "faq",
                "publisher": "eu_commission",
            }
        ]

    def test_does_not_rechunk_or_reembed(self, tmp_path: Path) -> None:
        # 切块与向量只依赖正文，元数据改了重跑一遍纯属白干
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), publisher=None, doc_id="doc-1")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        index_document(make_document(raw, publisher="eu_commission"), context)

        assert context.mysql.chunks_written == []
        assert context.milvus.upserted == []
        assert context.milvus.deleted == []

    def test_leaves_the_vector_store_when_metadata_breaks(self, tmp_path: Path) -> None:
        # 改完反而不齐了（比如把 publisher 清空）：门禁对它的要求和
        # 第一次入库时一样，不该继续留在向量库里
        raw = write_html_file(tmp_path)
        existing = make_record(file_content_hash(raw), doc_id="doc-1")
        context = make_context(mysql=RecordingMysqlStore(by_hash=existing))

        outcome = index_document(make_document(raw, publisher=None), context)

        assert context.milvus.deleted == ["doc-1"]
        assert ("doc-1", DocumentStatus.PENDING) in context.mysql.status_updates
        assert context.milvus.scalar_updates == []
        assert "publisher" in outcome.detail


class TestNewDocumentThroughIndexDocument:
    def test_registers_and_indexes(self, tmp_path: Path) -> None:
        context = make_context()

        outcome = index_document(make_document(write_html_file(tmp_path)), context)

        assert outcome.action is ImportAction.NEW
        assert outcome.ok is True
        assert len(context.mysql.inserted) == 1
        assert len(context.mysql.chunks_written) == 1
        assert context.milvus.upserted != []

    def test_outcome_matches_what_was_written(self, tmp_path: Path) -> None:
        context = make_context()

        outcome = index_document(make_document(write_html_file(tmp_path)), context)

        assert outcome.doc_id == context.mysql.inserted[0].doc_id
        assert outcome.file_name == "amazon-eu-vat-faq.html"
        assert outcome.parent_count >= 1
        assert outcome.child_count >= 1

    def test_metadata_gate_still_reports_ok(self, tmp_path: Path) -> None:
        # 没入库不算失败，文档确实收下了
        context = make_context()

        outcome = index_document(
            make_document(write_html_file(tmp_path), publisher=None), context
        )

        assert outcome.ok is True
        assert outcome.parent_count == 0
        assert "publisher" in outcome.detail
