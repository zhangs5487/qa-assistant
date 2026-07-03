"""Local Qwen3 embedding provider — Qwen3-Embedding-4B via transformers.

Qwen3-Embedding uses the encoder to produce dense text embeddings (dense retrieval).
Loaded via AutoModel, NOT AutoModelForCausalLM.
"""

import logging
from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from ..base import EmbeddingProvider

logger = logging.getLogger(__name__)


class Qwen3Embedding(EmbeddingProvider):
    """Embedding provider backed by Qwen3-Embedding-4B."""

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
    ):
        self._model_path = model_path
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        logger.info("Loading Qwen3-Embedding from %s on %s ...", self._model_path, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            self._model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=True,
        )
        if self._device == "cpu":
            self._model = self._model.to("cpu")
        self._model.eval()
        logger.info("Qwen3-Embedding loaded. dim=%s", self.embedding_dimensions)

    def embed_query(self, text: str) -> list[float]:
        self._load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS pooling
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0].tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._load()
        results = []
        for text in texts:
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :]
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            results.append(emb[0].tolist())
        return results

    @property
    def embedding_dimensions(self) -> int:
        self._load()
        return self._model.config.hidden_size
