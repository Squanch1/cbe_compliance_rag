"""按语料清单收集原始文档。

把清单里的 file_name 解析成 data/raw/ 下的真实路径，检查文件是否存在，
并为每条分配 doc_id、写入采集日期，最终产出 DocumentMeta。

缺文件的条目**不会被静默丢掉**，而是单独列出来交给调用方决定怎么处理。
手工维护清单时「先填行、后下文件」是常态，因此跳过是合理的，
但必须让使用者看见跳过了什么。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from cbe_rag.ingestion.fetcher.manifest import ManifestRow, load_manifest
from cbe_rag.ingestion.parser.schema import DocumentMeta, new_doc_id


class CollectionError(Exception):
    """收集无法继续时抛出。"""


@dataclass(frozen=True)
class CollectedDocument:
    """一份采集到的原始文档：文件位置加业务元数据。"""

    raw_path: Path
    meta: DocumentMeta


@dataclass(frozen=True)
class MissingFile:
    """清单引用了但文件不存在的一条。"""

    line_number: int
    file_name: str
    expected_path: Path


@dataclass(frozen=True)
class CollectionResult:
    """一次收集的完整结果。

    分成两份而不是合成一份，是为了让调用方能明确区分
    「可以进入解析流程的」与「还差文件的」。
    """

    documents: list[CollectedDocument]
    missing_files: list[MissingFile]


def _to_meta(item: ManifestRow, collected_date: date) -> DocumentMeta:
    """把清单行转成文档元数据，并分配 doc_id。"""
    return DocumentMeta(
        doc_id=new_doc_id(),
        title=item.title,
        collected_date=collected_date,
        source_url=item.source_url,
        publisher=item.publisher,
        country=item.country,
        doc_type=item.doc_type,
        effective_date=item.effective_date,
        platform=item.platform,
    )


def collect_documents(
    manifest_path: Path,
    raw_dir: Path,
    *,
    today: date | None = None,
) -> CollectionResult:
    """按清单收集原始文档。

    today 可注入，是为了让测试结果不随运行日期变化；
    生产路径下不传，取当天日期。

    清单本身的格式错误会直接抛出，不做「跳过这行继续」的处理——
    格式错误是笔误，静默跳过会让笔误一直留在表里。
    """
    if not raw_dir.is_absolute():
        raise CollectionError("raw_dir 必须是绝对路径，收到：%s" % raw_dir)
    if not raw_dir.is_dir():
        # 目录整个不存在时，逐条报「文件缺失」会淹没真正的原因
        raise CollectionError("原始文档目录不存在：%s" % raw_dir)

    collected_date = today if today is not None else date.today()

    documents: list[CollectedDocument] = []
    missing_files: list[MissingFile] = []

    for item in load_manifest(manifest_path):
        # file_name 已在清单校验中确认不含路径分隔符，因此这里只是拼接
        path = raw_dir / item.file_name
        if path.is_file():
            documents.append(
                CollectedDocument(
                    raw_path=path, meta=_to_meta(item, collected_date)
                )
            )
        else:
            missing_files.append(
                MissingFile(
                    line_number=item.line_number,
                    file_name=item.file_name,
                    expected_path=path,
                )
            )

    return CollectionResult(documents=documents, missing_files=missing_files)
