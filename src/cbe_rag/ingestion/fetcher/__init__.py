"""读取本地原始文档，登记来源与采集时间。"""

from cbe_rag.ingestion.fetcher.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    ManifestRow,
    load_manifest,
)

__all__ = [
    "MANIFEST_COLUMNS",
    "ManifestError",
    "ManifestRow",
    "load_manifest",
]
