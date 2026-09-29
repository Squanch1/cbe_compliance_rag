"""解析为统一中间表示。

对外分两组能力：

- **单格式解析器**：`parse_html` / `parse_pdf`，指定格式直接调用
- **路由入口**：`parse_document`，按文件头判类型、按难度组装解析链、
  逐层解析并做质量评估。日常调用走这个
"""

from cbe_rag.ingestion.parser.difficulty import (
    DifficultyLevel,
    DifficultyReport,
    probe_difficulty,
)
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
    body_font_size,
    build_blocks,
    extract_lines,
    heading_levels,
    parse_pdf,
)
from cbe_rag.ingestion.parser.quality import (
    QualityReport,
    QualityThresholds,
    assess,
)
from cbe_rag.ingestion.parser.router import (
    ParseRouteError,
    build_chain,
    detect_format,
    parse_document,
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
from cbe_rag.ingestion.parser.tier import (
    ParseOutcome,
    ParseRequest,
    TierAttempt,
    TierParser,
)

__all__ = [
    "HTML_PARSER_VERSION",
    "PDF_PARSER_VERSION",
    "SITE_CONTENT_SELECTORS",
    "Block",
    "BlockType",
    "Chunk",
    "ChunkLevel",
    "DifficultyLevel",
    "DifficultyReport",
    "DocumentMeta",
    "HtmlParseError",
    "ParseOutcome",
    "ParseRequest",
    "ParseRouteError",
    "ParsedDocument",
    "PdfParseError",
    "QualityReport",
    "QualityThresholds",
    "SourceFormat",
    "TextLine",
    "TierAttempt",
    "TierParser",
    "assess",
    "body_font_size",
    "build_blocks",
    "build_chain",
    "content_selector_for",
    "detect_format",
    "extract_blocks",
    "extract_lines",
    "heading_levels",
    "make_child_chunk_id",
    "make_parent_chunk_id",
    "new_doc_id",
    "parse_document",
    "parse_html",
    "parse_pdf",
    "probe_difficulty",
]
