"""Ollama provider implementations for local models.

Uses Ollama's REST API. No SDK needed — just ``httpx``.

Start Ollama: ``ollama serve``
Pull a model:  ``ollama pull nomic-embed-text``
"""

from typing import Iterator, Sequence

import httpx

from ..base import ChatConfig, ChatMessage, ChatProvider, EmbeddingProvider


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider backed by Ollama (nomic-embed-text, bge-m3, etc.)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=60.0)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        results = []
        for text in texts:
            resp = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            results.append(resp.json()["embedding"])
        return results

    @property
    def embedding_dimensions(self) -> int:
        # Query on first access
        try:
            emb = self.embed_query("dimension check")
            return len(emb)
        except Exception:
            return 768  # common default for nomic-embed-text


class OllamaChat(ChatProvider):
    """Chat provider backed by Ollama (llama3, qwen, mistral, etc.)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=120.0)

    def _build_messages(self, messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: list[ChatMessage], config: ChatConfig | None = None) -> str:
        cfg = config or ChatConfig()
        resp = self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": cfg.model or self._model,
                "messages": self._build_messages(messages),
                "stream": False,
                "options": {
                    "temperature": cfg.temperature,
                    "num_predict": cfg.max_tokens,
                    "top_p": cfg.top_p,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def chat_stream(
        self, messages: list[ChatMessage], config: ChatConfig | None = None
    ) -> Iterator[str]:
        cfg = config or ChatConfig()
        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": cfg.model or self._model,
                "messages": self._build_messages(messages),
                "stream": True,
                "options": {
                    "temperature": cfg.temperature,
                    "num_predict": cfg.max_tokens,
                    "top_p": cfg.top_p,
                },
            },
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = httpx.json.loads(line)
                if data.get("message", {}).get("content"):
                    yield data["message"]["content"]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 2)
