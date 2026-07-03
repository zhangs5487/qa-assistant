"""Local BGE embedding provider — BGE-M3 from ModelScope / HuggingFace.

Zero API cost, no rate limits, data stays local.

Requires ``pip install sentence-transformers``.

Model download (first use):
- HuggingFace: ``BAAI/bge-m3`` (~2.2 GB)
- ModelScope: ``modelscope/BAAI/bge-m3`` (set MODEL_NAME accordingly)

Usage:
    Set ``EMBEDDING_PROVIDER=local_bge`` and ``EMBEDDING_MODEL=BAAI/bge-m3`` in .env
"""

from typing import Sequence

from ..base import EmbeddingProvider


class BGEEmbedding(EmbeddingProvider):
    """Embedding provider backed by BGE-M3 (BAAI General Embedding).

    Loads the model locally via ``sentence-transformers``.  Supports:

    - Dense vector output (1024-dim for bge-m3)
    - Sparse lexical weights (via ``sparse_embedding`` method, for BM25-style retrieval)
    - CPU or CUDA auto-detection
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_dir: str = "./models",
        device: str | None = None,
    ):
        """
        Args:
            model_name: HuggingFace or ModelScope model identifier.
            cache_dir: Local directory to cache downloaded model weights.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._device = device
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(
                self._model_name,
                cache_folder=self._cache_dir,
                device=self._device,
            )
        return self._model

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        # BGE models benefit from a query instruction prefix
        if not text.startswith("为这个句子生成表示以用于检索相关文章："):
            text = "为这个句子生成表示以用于检索相关文章：" + text
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._get_model()
        # Documents should also use the instruction prefix for BGE
        prefixed = []
        for t in texts:
            if not t.startswith("为这个句子生成表示以用于检索相关文章："):
                prefixed.append("为这个句子生成表示以用于检索相关文章：" + t)
            else:
                prefixed.append(t)
        embeddings = model.encode(prefixed, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]

    @property
    def embedding_dimensions(self) -> int:
        model = self._get_model()
        dim = model.get_sentence_embedding_dimension()
        return dim or 1024

    def sparse_embedding(self, text: str) -> dict[int, float]:
        """Return sparse lexical weights (BGE-M3 specific).

        These can be used directly as BM25-style keyword scores,
        avoiding the need for a separate BM25 index in simple setups.

        Returns:
            Dict mapping token IDs to their lexical weights.
        """
        model = self._get_model()
        if not hasattr(model, "sparse_embedding"):
            raise NotImplementedError(
                "The loaded model does not support sparse embeddings. "
                "Use BGE-M3 (BAAI/bge-m3) for dense+sparse dual output."
            )
        result = model.sparse_embedding(text)
        return result
