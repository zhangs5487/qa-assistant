"""Abstract repository interfaces for the storage layer.

All database and file operations go through these interfaces.
Concrete implementations (MongoDB, PostgreSQL, MinIO, local FS)
live in separate modules and are swappable via config.
"""

from abc import ABC, abstractmethod

from shared.models import Chunk, CleanDocument, QAPair, RawDocument


class DocumentRepository(ABC):
    """CRUD operations for documents, chunks, and QA pairs."""

    @abstractmethod
    def save_raw(self, doc: RawDocument) -> str:
        """Persist a RawDocument. Returns the stored document ID."""
        ...

    @abstractmethod
    def save_clean(self, doc: CleanDocument) -> str:
        """Persist a CleanDocument. Returns the stored document ID."""
        ...

    @abstractmethod
    def save_chunks(self, chunks: list[Chunk]) -> int:
        """Batch-persist chunks. Returns the number inserted."""
        ...

    @abstractmethod
    def save_qa_pairs(self, pairs: list[QAPair]) -> int:
        """Batch-persist QA pairs. Returns the number inserted."""
        ...

    @abstractmethod
    def get_clean_by_url(self, url: str) -> CleanDocument | None:
        """Retrieve a CleanDocument by its source URL."""
        ...

    @abstractmethod
    def get_chunks_by_url(self, url: str) -> list[Chunk]:
        """Retrieve all chunks belonging to a document URL."""
        ...

    @abstractmethod
    def url_exists(self, url: str) -> bool:
        """Check whether a URL has already been crawled and stored."""
        ...

    @abstractmethod
    def count_documents(self) -> int:
        """Return the total number of stored documents."""
        ...


class FileStore(ABC):
    """Binary file storage — raw HTML, PDF, images, etc."""

    @abstractmethod
    def save(self, key: str, content: bytes, content_type: str = "") -> str:
        """Save raw bytes. Return the storage path or key."""
        ...

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load raw bytes by key."""
        ...

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a key exists."""
        ...

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a stored object."""
        ...


class VectorStore(ABC):
    """Vector database for embedding-based search.

    Used in Phase 2 (indexing / retrieval). Defined now so the interface
    is stable and the data pipeline knows what metadata to preserve.
    """

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create (or ensure existence of) a named collection."""
        ...

    @abstractmethod
    def insert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert vectors with associated metadata.

        Args:
            collection: Target collection name.
            ids: Unique identifiers (e.g., chunk IDs).
            embeddings: Vector arrays (one per id).
            metadata: Metadata dicts (one per id).
        """
        ...

    @abstractmethod
    def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Query the collection for nearest neighbours.

        Returns a list of dicts with keys: id, score, metadata.
        """
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        """Drop a collection entirely."""
        ...
