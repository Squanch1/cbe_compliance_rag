"""解析为统一中间表示。"""

from cbe_rag.ingestion.parser.html_parser import (
    SITE_CONTENT_SELECTORS,
    HtmlParseError,
    content_selector_for,
    extract_blocks,
)
from cbe_rag.ingestion.parser.schema import (
    Block,
    BlockType,
    Chunk,
    ChunkLevel,
    DocumentMeta,
    ParsedDocument,
    SourceFormat,
    make_child_chunk_id,
    make_parent_chunk_id,
    new_doc_id,
)

__all__ = [
    "SITE_CONTENT_SELECTORS",
    "Block",
    "BlockType",
    "Chunk",
    "ChunkLevel",
    "DocumentMeta",
    "HtmlParseError",
    "ParsedDocument",
    "SourceFormat",
    "content_selector_for",
    "extract_blocks",
    "make_child_chunk_id",
    "make_parent_chunk_id",
    "new_doc_id",
]
