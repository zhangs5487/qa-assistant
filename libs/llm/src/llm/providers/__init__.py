"""Provider implementations — import what you need or use the factory instead."""

from .local_bge import BGEEmbedding
from .openai import OpenAIChat, OpenAIEmbedding

__all__ = [
    # Local
    "BGEEmbedding",
    # OpenAI
    "OpenAIEmbedding",
    "OpenAIChat",
]
