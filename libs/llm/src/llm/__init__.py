"""LLM abstraction layer — provider-agnostic interfaces for embeddings and chat.

No code outside this package should import a provider SDK directly.
Use ``create_embedding_provider()`` and ``create_chat_provider()`` from ``factory.py``.
"""

from .base import ChatConfig, ChatMessage, ChatProvider, EmbeddingConfig, EmbeddingProvider
from .factory import create_chat_provider, create_embedding_provider

__all__ = [
    # Abstract base classes
    "EmbeddingProvider",
    "ChatProvider",
    # Config / data
    "EmbeddingConfig",
    "ChatConfig",
    "ChatMessage",
    # Factory
    "create_embedding_provider",
    "create_chat_provider",
]
