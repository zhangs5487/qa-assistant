"""Re-ranker module — cross-encoder-based re-ranking for improved retrieval quality.

Supports:
- BGE-Reranker-v2-m3 (via sentence-transformers CrossEncoder)
- Qwen3-Reranker-4B (via transformers AutoModelForSequenceClassification)

Usage:
    reranker = Reranker(model_path="./models/bge-reranker-v2-m3")
    results = reranker.rerank(query, candidates, top_k=5)
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker.

    Auto-detects model type: BGE-Reranker (sentence-transformers) or
    Qwen3-Reranker (transformers). Takes a query and candidate documents,
    scores each (query, doc) pair through the cross-encoder, and returns
    the top-k re-ranked results.
    """

    def __init__(self, model_path: str = "", device: str | None = None):
        """
        Args:
            model_path: Local path to the reranker model directory.
                        If empty, defaults to ``./models/bge-reranker-v2-m3``.
            device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
        """
        self._model_path = model_path or "./models/bge-reranker-v2-m3"
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._model_type = None  # "bge" or "qwen3"
        self._max_length = 512

    def _detect_model_type(self):
        """Detect whether the model is BGE-Reranker or Qwen3-Reranker by
        checking the config.json for architecture type."""
        import json
        from pathlib import Path

        config_path = Path(self._model_path) / "config.json"
        if not config_path.exists():
            logger.warning("Cannot find config.json in %s, assuming BGE type", self._model_path)
            return "bge"

        config = json.loads(config_path.read_text())
        archs = config.get("architectures", [])
        for a in archs:
            if "qwen" in a.lower():
                return "qwen3"
        return "bge"

    def _get_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model, self._tokenizer

        self._model_type = self._detect_model_type()
        logger.info("Loading %s reranker from %s ...", self._model_type, self._model_path)

        if self._model_type == "qwen3":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            if self._device == "cpu":
                self._model = self._model.to("cpu")
            self._model.eval()
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self._model_path,
                max_length=self._max_length,
                device=self._device,
            )
            self._tokenizer = None  # CrossEncoder handles its own tokenization

        logger.info("Reranker model loaded (type=%s).", self._model_type)
        return self._model, self._tokenizer

    def _score_pairs_qwen3(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score (query, doc) pairs using Qwen3-Reranker transformers model."""
        scores = []
        for query, doc in pairs:
            inputs = self._tokenizer(
                query, doc,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
                padding=True,
            ).to(self._model.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                # Binary classification: logits[0] = negative, logits[1] = positive
                score = torch.softmax(logits, dim=-1)[0, 1].item()
            scores.append(score)
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        text_key: str = "",
    ) -> list[dict[str, Any]]:
        """Re-rank candidate documents by cross-encoder relevance scoring.

        Args:
            query: The original user query.
            candidates: List of candidate result dicts.
            top_k: Number of top results to return after re-ranking.
            text_key: Dict key for the candidate text to score.
                      Auto-detected: tries ``"content"``, ``"question"``, ``"text"``.

        Returns:
            Re-ranked candidates with ``rerank_score`` field added,
            sorted by ``rerank_score`` descending, limited to ``top_k``.
        """
        if not candidates:
            return []

        model, tokenizer = self._get_model()

        # Auto-detect text key
        if not text_key:
            for key in ("content", "question", "text"):
                if key in candidates[0]:
                    text_key = key
                    break
            if not text_key:
                logger.warning("Cannot determine text key for reranking; using 'content'")
                text_key = "content"

        # Prepare (query, doc) pairs
        pairs = []
        valid_candidates = []
        for c in candidates:
            doc_text = c.get(text_key, "")
            if doc_text and len(doc_text) > 5:
                pairs.append((query, doc_text))
                valid_candidates.append(c)

        if not pairs:
            return candidates[:top_k]

        # Score
        try:
            if self._model_type == "qwen3":
                scores = self._score_pairs_qwen3(pairs)
            else:
                scores = model.predict(pairs)
        except Exception as e:
            logger.warning("Reranker prediction failed, falling back: %s", e)
            return candidates[:top_k]

        # Attach reranker scores and sort
        for i, c in enumerate(valid_candidates):
            c["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0

        reranked = sorted(valid_candidates, key=lambda x: x["rerank_score"], reverse=True)
        logger.debug("Reranked %d -> top %d (best: %.4f)", len(valid_candidates), top_k,
                     reranked[0]["rerank_score"] if reranked else 0)
        return reranked[:top_k]
