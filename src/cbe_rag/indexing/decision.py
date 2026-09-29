"""判重决策。

输入是几个现成的事实——按哈希查到的记录、按 URL 查到的活跃记录，加上
清单上登记的元数据——输出是一个动作。**不查库也不写库**，因此规则本身
可以脱离真实存储来测。判重是最容易出错的一环：判错的后果不是崩溃，而是
同一份文档重复入库或悄悄不再被检索到，两种都不报错。

规则见 docs/spec/03-data-model.md 3.2。
"""

from __future__ import annotations

from cbe_rag.ingestion.parser.schema import DocumentMeta
from cbe_rag.indexing.models import Decision, ImportAction
from cbe_rag.storage.ddl import DocumentStatus
from cbe_rag.storage.records import DocumentRecord

# 判重时要比对的元数据字段。
#
# 不含 doc_id（两份文件的标识本就不同）与 collected_date（采集日期，
# 每次导入都会变，拿它比对等于每次都判为「改过」）。
#
# 有测试断言这里恰好是 DocumentMeta 除那两个字段之外的全部，将来给
# DocumentMeta 加字段时测试会失败，提醒决定它要不要参与比对。
_COMPARED_FIELDS: tuple[str, ...] = (
    "title",
    "source_url",
    "publisher",
    "country",
    "doc_type",
    "effective_date",
    "platform",
)


def decide(
    meta: DocumentMeta,
    *,
    by_hash: DocumentRecord | None = None,
    by_url: DocumentRecord | None = None,
) -> Decision:
    """判定这份文档该怎么处理。

    by_hash 是按内容哈希查到的记录；by_url 是按 source_url 查到的
    **活跃**记录，已下线的那些不必查，它们本就不参与检索。

    内容命中优先于 URL 命中：哈希相同说明文件一个字节都没变，这时
    同一链接下还有没有别的记录，不影响本次处理。
    """
    if by_hash is not None:
        # 非 indexed 的记录说明上一次没走完——元数据不齐、解析各层都
        # 不合格、中途出错，或者被新版本取代过。这几种情况原先的结论
        # 已经不作数了，必须重跑才能改变结果，不能因为内容相同就跳过。
        # 漏掉这条的后果是文档悄悄停在不可检索的状态，而报告显示「已跳过」。
        if by_hash.status is not DocumentStatus.INDEXED:
            return Decision(ImportAction.REINDEX, matched=by_hash)

        # 内容相同，再看元数据有没有改过（例如补上了 publisher）
        if any(
            getattr(meta, name) != getattr(by_hash, name)
            for name in _COMPARED_FIELDS
        ):
            return Decision(ImportAction.UPDATE_META, matched=by_hash)

        return Decision(ImportAction.SKIP, matched=by_hash)

    if by_url is not None:
        # 同一链接下的内容变了，是这份文档的新版本。旧记录保留可追溯
        # 版本变更，但要下线——旧内容被召回会给出过时答案，而且答案
        # 下面还挂着引用，看起来有出处。
        return Decision(ImportAction.SUPERSEDE_AND_NEW, previous=by_url)

    return Decision(ImportAction.NEW)
