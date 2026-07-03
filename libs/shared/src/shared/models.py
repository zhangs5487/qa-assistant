"""Core data contracts shared across all packages.

Every package imports these models. No package should import another
package's internals — only what is defined here.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from .enums import DocumentSource, DocumentStatus


# ---------------------------------------------------------------------------
# Crawler output
# ---------------------------------------------------------------------------

class RawDocument(BaseModel):
    """One raw document emitted by the crawler.

    Represents a single crawled page or file before any processing.
    """

    url: str
    source: DocumentSource
    raw_content: str | bytes  # HTML string, PDF bytes, or plain text
    content_type: str  # MIME type: "text/html", "application/pdf", "text/plain"
    title: str | None = None
    category: str | None = None  # Inferred from URL path or page metadata
    crawl_time: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict = Field(default_factory=dict)  # HTTP headers, fetch duration, etc.


# ---------------------------------------------------------------------------
# Cleaner input / output
# ---------------------------------------------------------------------------

class QAPair(BaseModel):
    """A single question–answer pair extracted from crawled content."""

    question: str
    answer: str
    source_url: str | None = None
    source_doc_id: str | None = None  # Links back to a CleanDocument
    confidence: float | None = None  # 0–1 when extracted heuristically


class CleanDocument(BaseModel):
    """A parsed and cleaned document ready for chunking or QA extraction."""

    source_url: str
    original_title: str | None = None
    clean_title: str | None = None
    clean_content: str  # Plain text — no HTML, no ads, no navigation chrome
    content_length_chars: int = 0
    category: str | None = None
    language: str = "zh"  # ISO 639-1
    extracted_qas: list[QAPair] = Field(default_factory=list)
    source: DocumentSource = DocumentSource.UNKNOWN
    status: DocumentStatus = DocumentStatus.CLEANED
    processed_time: datetime = Field(default_factory=datetime.utcnow)
    cleaning_meta: dict = Field(default_factory=dict)  # Which cleaners ran, encoding info, etc.


class Chunk(BaseModel):
    """One contiguous chunk of a CleanDocument, ready for embedding."""

    chunk_id: str = Field(default_factory=lambda: uuid4().hex)
    document_url: str
    chunk_index: int  # 0-based position within the document
    content: str
    token_count: int | None = None
    overlap_with_prev: bool = False
    metadata: dict = Field(default_factory=dict)
