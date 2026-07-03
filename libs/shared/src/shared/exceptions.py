"""Pipeline-level exceptions shared across packages."""


class PipelineError(Exception):
    """Base exception for pipeline failures."""


class CrawlError(PipelineError):
    """Raised when crawling fails for a specific URL or domain."""


class ParseError(PipelineError):
    """Raised when parsing a document fails (e.g., corrupt HTML, unreadable PDF)."""


class CleanError(PipelineError):
    """Raised when cleaning/normalization fails."""


class ChunkError(PipelineError):
    """Raised when chunking fails (e.g., empty content, invalid parameters)."""


class StorageError(PipelineError):
    """Raised when a storage operation fails (DB connection, write failure)."""


class ConfigurationError(PipelineError):
    """Raised when required configuration is missing or invalid."""
