"""Provider implementations — import what you need or use the factory instead."""

from .local_bge import BGEEmbedding
from .ollama import OllamaChat, OllamaEmbedding
from .openai import OpenAIChat, OpenAIEmbedding

__all__ = [
    # Local
    "BGEEmbedding",
    # OpenAI
    "OpenAIEmbedding",
    "OpenAIChat",
    # Ollama
    "OllamaEmbedding",
    "OllamaChat",
]
