"""解析为统一中间表示。"""

from cbe_rag.ingestion.parser.html_parser import (
    HTML_PARSER_VERSION,
    SITE_CONTENT_SELECTORS,
    HtmlParseError,
    content_selector_for,
    extract_blocks,
    parse_html,
)
from cbe_rag.ingestion.parser.pdf_parser import (
    PDF_PARSER_VERSION,
    PdfParseError,
    TextLine,
    extract_lines,
    parse_pdf,
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
    "HTML_PARSER_VERSION",
    "SITE_CONTENT_SELECTORS",
    "Block",
    "BlockType",
    "Chunk",
    "ChunkLevel",
    "DocumentMeta",
    "HtmlParseError",
    "PDF_PARSER_VERSION",
    "ParsedDocument",
    "PdfParseError",
    "SourceFormat",
    "TextLine",
    "content_selector_for",
    "extract_lines",
    "extract_blocks",
    "make_child_chunk_id",
    "make_parent_chunk_id",
    "new_doc_id",
    "parse_html",
    "parse_pdf",
]
