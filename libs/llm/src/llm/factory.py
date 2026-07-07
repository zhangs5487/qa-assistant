"""Factory functions for creating LLM provider instances.

Reads provider type and credentials from the global Settings object.
"""

from config.settings import settings

from .base import ChatProvider, EmbeddingProvider


def create_embedding_provider() -> EmbeddingProvider:
    """Return an EmbeddingProvider based on current config.

    Reads ``settings.embedding_provider`` to decide which implementation
    to instantiate.  Valid values:

    - ``local_bge``          — BGE-M3 (sentence-transformers)
    - ``local_qwen3``        — Qwen3-Embedding-4B (transformers)
    - ``openai``             — OpenAI Embedding API
    - ``api``                — OpenAI-compatible API (uses chat credentials)

    Raises:
        ValueError: If the configured provider is unknown.
        ImportError: If the required SDK is not installed for the chosen provider.
    """
    provider_name = settings.embedding_provider

    if provider_name == "local_bge":
        from .providers.local_bge import BGEEmbedding

        return BGEEmbedding(
            model_name=settings.embedding_model or "BAAI/bge-m3",
            cache_dir=settings.models_cache_dir or "./models",
            device=None,
        )

    elif provider_name == "local_qwen3":
        from .providers.local_qwen_embedding import Qwen3Embedding

        return Qwen3Embedding(
            model_path=settings.embedding_model or "./models/qwen3-embeding-4b",
        )

    elif provider_name == "openai":
        from .providers.openai import OpenAIEmbedding

        return OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model=settings.embedding_model or "text-embedding-3-small",
            base_url=settings.openai_base_url or None,
        )

    elif provider_name == "api":
        from .providers.openai import OpenAIEmbedding

        if not settings.chat_api_key:
            raise ValueError(
                "EMBEDDING_PROVIDER=api 但 CHAT_API_KEY 未配置。"
                "请在 .env 中设置 CHAT_API_KEY 和 CHAT_API_BASE_URL。"
            )
        return OpenAIEmbedding(
            api_key=settings.chat_api_key,
            model=settings.embedding_model or "text-embedding-3-small",
            base_url=settings.chat_api_base_url or None,
        )

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name!r}. "
            "Valid options: local_bge, local_qwen3, openai, api"
        )


def create_chat_provider() -> ChatProvider:
    """Return a ChatProvider based on current config.

    Reads ``settings.chat_provider`` to decide which implementation
    to instantiate.  Valid values: ``api``, ``openai``, ``local_llm``.

    Raises:
        ValueError: If the configured provider is unknown.
        ImportError: If the required SDK is not installed for the chosen provider.
    """
    provider_name = settings.chat_provider

    if provider_name == "api":
        from .providers.openai import OpenAIChat

        if not settings.chat_api_key:
            raise ValueError(
                "CHAT_PROVIDER=api 但 CHAT_API_KEY 未配置。"
                "请在 .env 中设置 CHAT_API_KEY 和 CHAT_API_BASE_URL。"
            )
        return OpenAIChat(
            api_key=settings.chat_api_key,
            model=settings.chat_model or "deepseek-v4-flash",
            base_url=settings.chat_api_base_url or None,
        )

    elif provider_name == "openai":
        from .providers.openai import OpenAIChat

        return OpenAIChat(
            api_key=settings.openai_api_key,
            model=settings.chat_model or "gpt-4o-mini",
            base_url=settings.openai_base_url or None,
        )

    elif provider_name == "local_llm":
        from .providers.local_qwen import LocalQwenChat

        return LocalQwenChat(
            model_path=settings.local_llm_path,
        )

    else:
        raise ValueError(
            f"Unknown chat provider: {provider_name!r}. "
            "Valid options: api, openai, local_llm"
        )
