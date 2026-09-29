"""索引流程用到的数据结构。

三组：

- `ImportAction`：判重得出的四种动作
- `Decision`：判重的完整结果
- `DocumentOutcome` / `ImportReport`：执行结果与批量汇总

判重的规则本身在 decision.py，这里只有数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cbe_rag.ingestion.fetcher.collector import MissingFile
from cbe_rag.storage.records import DocumentRecord


class ImportAction(str, Enum):
    """判重得出的动作。"""

    # 没见过这份文件，走完整的解析与索引，生成新的 doc_id
    NEW = "new"
    # 内容见过，但库里那条没走完（元数据不齐、解析各层不合格、中途出错、
    # 或被新版本取代过），复用它的 doc_id 重跑一遍
    REINDEX = "reindex"
    # 内容见过且已入库，但清单上的元数据变了（如补上了 publisher），只改属性
    UPDATE_META = "update_meta"
    # 内容与元数据都一致，什么都不做
    SKIP = "skip"
    # 同一 source_url 下出了新内容，旧记录下线，新内容走完整流程
    SUPERSEDE_AND_NEW = "supersede_and_new"


@dataclass(frozen=True)
class Decision:
    """判重的完整结果。

    matched 是内容命中的那条记录（SKIP 与 UPDATE_META 时有值），
    doc_id 要从它身上取，不能另生成。

    previous 是同 source_url 的旧记录，内容已经变了（SUPERSEDE_AND_NEW
    时有值），它将被标记下线。
    """

    action: ImportAction
    matched: DocumentRecord | None = None
    previous: DocumentRecord | None = None


@dataclass(frozen=True)
class DocumentOutcome:
    """一份文档处理完的结果。

    detail 是给人看的一句话：成功时说清做了什么，失败时说清卡在哪。
    """

    file_name: str
    action: ImportAction
    ok: bool
    detail: str
    doc_id: str | None = None
    parent_count: int = 0
    child_count: int = 0


@dataclass(frozen=True)
class ImportReport:
    """一次批量导入的完整结果。

    orphans 是「库里有、清单里没有」的文档。**只报告不动手**：删除
    不可逆，且清单改错一个字就会让文档从检索里消失，因此由人看过
    清单再决定。
    """

    outcomes: list[DocumentOutcome]
    missing_files: list[MissingFile]
    orphans: list[DocumentRecord]


def count_by_action(report: ImportReport, action: ImportAction) -> int:
    """统计某个动作发生了多少次。

    报告要打印各类动作的条数，调用方不必自己遍历一遍。
    """
    return sum(1 for outcome in report.outcomes if outcome.action is action)
