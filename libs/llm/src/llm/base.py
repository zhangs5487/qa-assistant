"""Abstract base classes for LLM provider abstraction.

Every provider implementation inherits from these. The rest of the codebase
programs to these interfaces — never to a specific provider.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatConfig:
    """Parameters for a chat completion request."""

    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    stream: bool = False


@dataclass
class EmbeddingConfig:
    """Parameters for an embedding request."""

    model: str = ""
    dimensions: int | None = None  # Some providers allow setting output dimensions


# ---------------------------------------------------------------------------
# Abstract providers
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract embedding provider — vectorize text for search and indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: A single query string (e.g., user question).

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of document strings.

        Args:
            texts: A sequence of document chunk strings.

        Returns:
            A list of embedding vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def embedding_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors produced."""
        ...

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Override for provider-specific tokenizers."""
        # Rough Chinese/English approximation: ~1.5 chars per token
        return max(1, len(text) // 2)


class ChatProvider(ABC):
    """Abstract chat/completion provider — generate answers from prompts."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage], config: ChatConfig | None = None) -> str:
        """Non-streaming chat completion.

        Args:
            messages: The conversation history / prompt.
            config: Optional generation parameters.

        Returns:
            The full assistant response as a string.
        """
        ...

    @abstractmethod
    def chat_stream(
        self, messages: list[ChatMessage], config: ChatConfig | None = None
    ) -> Iterator[str]:
        """Streaming chat completion.

        Yields tokens as they arrive from the provider.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the token count for the given text using the provider's tokenizer."""
        ...
