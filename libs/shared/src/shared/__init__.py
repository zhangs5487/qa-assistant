"""Shared data contracts — zero-dependency layer imported by all other packages."""

from .enums import DocumentSource, DocumentStatus
from .exceptions import (
    ChunkError,
    CleanError,
    ConfigurationError,
    CrawlError,
    ParseError,
    PipelineError,
    StorageError,
)
from .models import Chunk, CleanDocument, QAPair, RawDocument

__all__ = [
    # Models
    "RawDocument",
    "CleanDocument",
    "Chunk",
    "QAPair",
    # Enums
    "DocumentSource",
    "DocumentStatus",
    # Exceptions
    "PipelineError",
    "CrawlError",
    "ParseError",
    "CleanError",
    "ChunkError",
    "StorageError",
    "ConfigurationError",
]
