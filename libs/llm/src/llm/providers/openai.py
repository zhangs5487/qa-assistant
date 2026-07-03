"""OpenAI provider implementations.

Requires ``pip install openai``.

Also works with OpenAI-compatible APIs (Azure, local proxies, etc.)
by setting ``OPENAI_BASE_URL`` in config.
"""

from typing import Iterator, Sequence

from ..base import ChatConfig, ChatMessage, ChatProvider, EmbeddingProvider


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider backed by OpenAI (text-embedding-3-small, etc.)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai SDK is not installed. Run: pip install openai"
                )
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def embed_query(self, text: str) -> list[float]:
        resp = self._get_client().embeddings.create(
            model=self._model, input=text
        )
        return resp.data[0].embedding

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self._get_client().embeddings.create(
            model=self._model, input=list(texts)
        )
        return [d.embedding for d in resp.data]

    @property
    def embedding_dimensions(self) -> int:
        if "3-large" in self._model:
            return 3072
        return 1536  # text-embedding-3-small / ada-002


class OpenAIChat(ChatProvider):
    """Chat provider backed by OpenAI (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str | None = None):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai SDK is not installed. Run: pip install openai"
                )
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[ChatMessage], config: ChatConfig | None = None) -> str:
        cfg = config or ChatConfig()
        resp = (
            self._get_client()
            .chat.completions.create(
                model=cfg.model or self._model,
                messages=self._build_messages(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
            )
        )
        return resp.choices[0].message.content

    def chat_stream(
        self, messages: list[ChatMessage], config: ChatConfig | None = None
    ) -> Iterator[str]:
        cfg = config or ChatConfig()
        resp = (
            self._get_client()
            .chat.completions.create(
                model=cfg.model or self._model,
                messages=self._build_messages(messages),
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                stream=True,
            )
        )
        for chunk in resp:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            return max(1, len(text) // 2)
