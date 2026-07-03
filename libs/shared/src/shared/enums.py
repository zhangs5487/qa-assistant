"""Shared enumerations used across all packages."""

from enum import Enum


class DocumentSource(str, Enum):
    """Identifies the origin of a document."""
    CQAIP_HTML = "cqaip_html"
    CQAIP_PDF = "cqaip_pdf"
    CQAIP_QA_PAIR = "cqaip_qa_pair"
    UNKNOWN = "unknown"


class DocumentStatus(str, Enum):
    """Tracks the processing state of a document through the pipeline."""
    RAW = "raw"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    CHUNKED = "chunked"
    INDEXED = "indexed"
    FAILED = "failed"
    SKIPPED = "skipped"
