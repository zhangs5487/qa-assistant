"""Document chunking using LangChain text splitters.

Splits CleanDocument content into smaller chunks for RAG embedding.
"""

import logging
from typing import Sequence

from shared.models import Chunk, CleanDocument

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Splits CleanDocuments into overlapping text chunks.

    Uses LangChain's RecursiveCharacterTextSplitter with
    Chinese-aware separators.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens (approximate).
            chunk_overlap: Number of tokens overlapping between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = None

    def _get_splitter(self):
        if self._splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
            except ImportError:
                raise ImportError(
                    "langchain-text-splitters not installed. "
                    "Run: pip install langchain-text-splitters"
                )
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=[
                    "\n\n",
                    "\n",
                    "。",
                    "！",
                    "？",
                    "；",
                    ".",
                    " ",
                    "",
                ],
            )
        return self._splitter

    def chunk_document(self, doc: CleanDocument) -> list[Chunk]:
        """Split a single CleanDocument into chunks.

        Args:
            doc: A cleaned document.

        Returns:
            List of Chunk objects.
        """
        if not doc.clean_content:
            return []

        splitter = self._get_splitter()
        texts = splitter.split_text(doc.clean_content)

        chunks = []
        for i, text in enumerate(texts):
            # Skip tiny fragments
            if len(text.strip()) < 10:
                continue
            chunk = Chunk(
                document_url=doc.source_url,
                chunk_index=i,
                content=text,
                token_count=len(text),  # rough char-based estimate
                overlap_with_prev=i > 0,
                metadata={
                    "title": doc.clean_title or "",
                    "category": doc.category or "",
                    "source": doc.source_url,
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_documents(self, docs: Sequence[CleanDocument]) -> list[Chunk]:
        """Split multiple documents into chunks.

        Args:
            docs: An iterable of CleanDocument objects.

        Returns:
            Combined list of Chunk objects.
        """
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
