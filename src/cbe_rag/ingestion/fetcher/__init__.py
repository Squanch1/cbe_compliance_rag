"""读取本地原始文档，登记来源与采集时间。"""

from cbe_rag.ingestion.fetcher.collector import (
    CollectedDocument,
    CollectionError,
    CollectionResult,
    MissingFile,
    collect_documents,
)
from cbe_rag.ingestion.fetcher.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRow,
    load_manifest,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "CollectedDocument",
    "CollectionError",
    "CollectionResult",
    "ManifestError",
    "ManifestRow",
    "MissingFile",
    "collect_documents",
    "load_manifest",
]
