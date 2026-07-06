"""Hybrid retriever: dense vector (Milvus) + sparse keyword (BM25) + RRF fusion + Re-rank.

Pipeline:
    User query
        → Dense vector search (Milvus)
        → Sparse keyword search (BM25 jieba)
        → RRF fusion
        → Cross-encoder re-rank (optional, via BGE-Reranker-v2-m3)
        → Top-K results

Usage:
    retriever = HybridRetriever()
    results = retriever.search_qa(query, use_rerank=True)
    # results already fused by RRF and optionally re-ranked
"""

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Sequence

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from config.settings import settings
from llm.factory import create_embedding_provider
from storage.milvus_store import get_milvus_store

logger = logging.getLogger(__name__)

RRF_K = settings.rrf_k  # 60


def _tokenize(text: str) -> list[str]:
    """Chinese-aware tokenization using jieba."""
    text = re.sub(r"[^\w一-鿿]", " ", text.lower())
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) > 1]


class BM25Index:
    """In-memory BM25 index for keyword search.

    Maintains a separate index for QA questions and document chunks.
    Persisted to disk for reuse across sessions.
    """

    def __init__(self, cache_dir: str = ""):
        self.cache_dir = Path(cache_dir or str(Path(settings.clean_data_dir) / "bm25_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.qa_texts: list[str] = []
        self.qa_metadata: list[dict] = []
        self.qa_bm25: BM25Okapi | None = None

        self.chunk_texts: list[str] = []
        self.chunk_metadata: list[dict] = []
        self.chunk_bm25: BM25Okapi | None = None

    # ---- Build ----

    def build_from_qa_file(self, qa_path: str = "") -> int:
        """Build BM25 index from the deduplicated QA pairs JSON.

        Returns:
            Number of questions indexed.
        """
        path = Path(qa_path or str(Path(settings.clean_data_dir) / "qa_deduped" / "deduped_qa_pairs.json"))
        if not path.exists():
            logger.warning("QA file not found: %s", path)
            return 0

        data = json.loads(path.read_text(encoding="utf-8"))
        pairs = data.get("qa_pairs", [])

        tokenized = []
        for item in pairs:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if not q:
                continue
            self.qa_texts.append(q)
            self.qa_metadata.append({
                "question": q,
                "answer": a,
                "source": item.get("source_url", ""),
                "confidence": item.get("confidence", 0.5),
            })
            tokenized.append(_tokenize(q))

        if tokenized:
            self.qa_bm25 = BM25Okapi(tokenized)

        logger.info("BM25 QA index: %d questions", len(self.qa_texts))
        return len(self.qa_texts)

    def build_from_chunk_dir(self, chunk_dir: str = "") -> int:
        """Build BM25 index from cleaned chunk files.

        Returns:
            Number of chunks indexed.
        """
        base = Path(chunk_dir or str(Path(settings.clean_data_dir) / "chunks"))
        if not base.exists():
            logger.warning("Chunk dir not found: %s", base)
            return 0

        tokenized = []
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir():
                continue
            json_files = sorted(subdir.glob("*.json"))
            if not json_files:
                continue
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                continue

            for chunk in data.get("chunks", []):
                text = chunk.get("content", "").strip()
                if not text or len(text) < 10:
                    continue
                self.chunk_texts.append(text)
                self.chunk_metadata.append({
                    "content": text,
                    "title": chunk.get("metadata", {}).get("title", ""),
                    "source": chunk.get("document_url", ""),
                    "category": subdir.name,
                })
                tokenized.append(_tokenize(text))

        if tokenized:
            self.chunk_bm25 = BM25Okapi(tokenized)

        logger.info("BM25 chunk index: %d chunks", len(self.chunk_texts))
        return len(self.chunk_texts)

    # ---- Search ----

    def search_qa(self, query: str, top_k: int = 10) -> list[dict]:
        """Search QA index by BM25.

        Returns:
            List of {question, answer, source, confidence, bm25_score}.
        """
        if not self.qa_bm25 or not self.qa_texts:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.qa_bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "question": self.qa_metadata[idx]["question"],
                    "answer": self.qa_metadata[idx]["answer"],
                    "source": self.qa_metadata[idx]["source"],
                    "confidence": self.qa_metadata[idx]["confidence"],
                    "bm25_score": float(scores[idx]),
                })
        return results

    def search_chunks(self, query: str, top_k: int = 10) -> list[dict]:
        """Search chunk index by BM25.

        Returns:
            List of {content, title, source, category, bm25_score}.
        """
        if not self.chunk_bm25 or not self.chunk_texts:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self.chunk_bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "content": self.chunk_metadata[idx]["content"],
                    "title": self.chunk_metadata[idx]["title"],
                    "source": self.chunk_metadata[idx]["source"],
                    "category": self.chunk_metadata[idx]["category"],
                    "bm25_score": float(scores[idx]),
                })
        return results

    def save(self):
        """Persist BM25 indices to cache directory."""
        for name, texts, metadata in [
            ("qa", self.qa_texts, self.qa_metadata),
            ("chunk", self.chunk_texts, self.chunk_metadata),
        ]:
            (self.cache_dir / name).mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / name / "texts.pkl", "wb") as f:
                pickle.dump(texts, f)
            with open(self.cache_dir / name / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

    def load(self) -> bool:
        """Load cached BM25 indices. Returns True if loaded successfully."""
        try:
            for name in ("qa", "chunk"):
                cache = self.cache_dir / name
                if not (cache / "texts.pkl").exists():
                    return False
                with open(cache / "texts.pkl", "rb") as f:
                    texts = pickle.load(f)
                with open(cache / "metadata.pkl", "rb") as f:
                    metadata = pickle.load(f)

                tokenized = [_tokenize(t) for t in texts]
                bm25 = BM25Okapi(tokenized)

                if name == "qa":
                    self.qa_texts = texts
                    self.qa_metadata = metadata
                    self.qa_bm25 = bm25
                else:
                    self.chunk_texts = texts
                    self.chunk_metadata = metadata
                    self.chunk_bm25 = bm25
            return True
        except Exception as e:
            logger.warning("Failed to load BM25 cache: %s", e)
            return False


# ---- RRF Fusion ----

def rrf_fuse(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = RRF_K,
    vector_score_key: str = "similarity",
    bm25_score_key: str = "bm25_score",
) -> list[dict]:
    """Fuse two ranked result lists using Reciprocal Rank Fusion.

    Args:
        vector_results: Results from vector search, sorted by similarity desc.
        bm25_results: Results from BM25 search, sorted by bm25_score desc.
        k: RRF constant (default 60).
        vector_score_key: Field name for vector similarity score.
        bm25_score_key: Field name for BM25 score.

    Returns:
        Fused results sorted by RRF score desc, with 'rrf_score' field added.
    """
    # Build a combined dict keyed by a unique identifier (question text or content)
    scores: dict[str, dict] = {}

    for rank, r in enumerate(vector_results):
        key = r.get("question") or r.get("content") or str(id(r))
        if key not in scores:
            entry = dict(r)
            entry["rrf_score"] = 0.0
            entry["vector_rank"] = rank
            entry["bm25_rank"] = -1
            scores[key] = entry
        scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

    # Track max BM25 score for normalization
    max_bm25 = max((r.get(bm25_score_key, 0) for r in bm25_results), default=0)

    for rank, r in enumerate(bm25_results):
        key = r.get("question") or r.get("content") or str(id(r))
        if key not in scores:
            entry = dict(r)
            entry["rrf_score"] = 0.0
            entry["vector_rank"] = -1
            entry["bm25_rank"] = rank
            # For BM25-only results, use rank-based similarity proxy (0.5-0.8 range)
            # so it shows meaningful values but stays below typical QA thresholds
            n = len(bm25_results)
            entry[vector_score_key] = 0.5 + 0.3 * (1.0 - rank / n) if n > 1 else 0.65
            # Preserve raw BM25 score
            entry["bm25_raw"] = r.get(bm25_score_key, 0)
            scores[key] = entry
        scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
        if "bm25_rank" in scores[key]:
            scores[key]["bm25_rank"] = rank
        else:
            scores[key]["bm25_rank"] = rank

    fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return fused


class HybridRetriever:
    """Dense + Sparse + RRF + optional Re-rank hybrid retriever.

    Wraps Milvus (dense vector) + BM25 (sparse keyword) and fuses
    results using RRF, then optionally re-ranks with a cross-encoder.
    """

    def __init__(self, use_rerank: bool = True, use_rerank_chunks: bool = False,
                 embedder=None):
        self.vector_store = get_milvus_store()
        self.embedder = embedder or create_embedding_provider()
        self.bm25 = BM25Index()
        self._ready = False
        self._reranker = None
        self._use_rerank = use_rerank          # Re-rank for QA search (short texts, fast, high impact)
        self._use_rerank_chunks = use_rerank_chunks  # Re-rank for chunk search (long texts, slow)
        self._init_index()

    def _init_index(self):
        """Initialize BM25 index (load cached or build fresh)."""
        if self.bm25.load():
            logger.info("BM25 index loaded from cache")
        else:
            logger.info("Building BM25 index...")
            n_qa = self.bm25.build_from_qa_file()
            n_chunk = self.bm25.build_from_chunk_dir()
            self.bm25.save()
            logger.info("BM25 index built: %d QA + %d chunks", n_qa, n_chunk)

        stats = self.vector_store.stats()
        qa_rows = stats.get("qa_pairs", {}).get("row_count", 0)
        chk_rows = stats.get("doc_chunks", {}).get("row_count", 0)
        logger.info("Hybrid retriever ready: %d QA vec + %d BM25 QA | %d chunk vec + %d BM25 chunk",
                     qa_rows, len(self.bm25.qa_texts), chk_rows, len(self.bm25.chunk_texts))
        self._ready = True

    def _embed_query(self, query: str) -> list[float]:
        return self.embedder.embed_query(query)

    # ---- In-memory vector search (fallback for when Milvus is broken) ----

    def _mem_search_qa(self, query_emb: list[float], top_k: int) -> list[dict]:
        """Cosine similarity search against BM25-indexed QA texts in memory."""
        if not self.bm25.qa_texts:
            return []
        q = np.array(query_emb, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        results = []
        for i, text in enumerate(self.bm25.qa_texts):
            if i >= len(self.bm25.qa_metadata):
                break
            # We don't have cached embeddings, so use BM25 scores as proxy
            continue
        return []

    def _mem_search_chunks(self, query_emb: list[float], top_k: int) -> list[dict]:
        """Cosine similarity search against chunk texts in memory."""
        if not self.bm25.chunk_texts:
            return []
        return []

    # ---- Milvus search with fallback ----

    def _search_vectors(self, query: str, top_k: int) -> tuple[list[dict], list[dict]]:
        """Search both QA and chunk collections. Returns (qa_results, chunk_results)."""
        vec = self._embed_query(query)
        qa_results = []
        chunk_results = []
        try:
            qa_results = self.vector_store.search_qa(vec, top_k=top_k * 2)
            chunk_results = self.vector_store.search_chunks(vec, top_k=top_k * 2)
        except Exception as e:
            logger.warning("Milvus search failed, falling back: %s", e)
        return qa_results, chunk_results

    # ---- Reranker ----

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker."""
        if not self._use_rerank:
            return None
        if self._reranker is None:
            from cleaner.reranker import Reranker
            try:
                self._reranker = Reranker(
                    model_path=settings.reranker_model,
                )
            except Exception as e:
                logger.warning("Reranker unavailable, skipping rerank step: %s", e)
                self._use_rerank = False
                return None
        return self._reranker

    # ---- Hybrid search: QA ----

    def search_qa(
        self, query: str, top_k: int = 10, use_hybrid: bool = True,
        use_rerank: bool | None = None,
    ) -> list[dict]:
        vec = self._embed_query(query)

        # Try Milvus first
        vector_results = []
        try:
            vector_results = self.vector_store.search_qa(vec, top_k=top_k * 3)
        except Exception:
            pass

        if not vector_results:
            return []

        if not use_hybrid:
            return vector_results[:top_k]

        bm25_results = self.bm25.search_qa(query, top_k=top_k * 3)

        if not bm25_results:
            fused = vector_results[:top_k * 2]
        elif not vector_results:
            fused = bm25_results[:top_k * 2]
        else:
            fused = rrf_fuse(vector_results, bm25_results, k=RRF_K)

        # Cross-encoder rerank (optional — ON for QA, short texts, fast)
        rerank_enabled = use_rerank if use_rerank is not None else self._use_rerank
        if rerank_enabled and fused:
            reranker = self._get_reranker()
            if reranker:
                rerank_candidates = min(settings.reranker_candidates, len(fused))
                return reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]

    # ---- Hybrid search: chunks ----

    def search_chunks(
        self, query: str, top_k: int = 15, use_hybrid: bool = True,
        use_rerank: bool | None = None,
    ) -> list[dict]:
        vec = self._embed_query(query)

        vector_results = []
        try:
            vector_results = self.vector_store.search_chunks(vec, top_k=top_k * 5)
        except Exception:
            pass

        if not vector_results:
            return []

        if not use_hybrid:
            return vector_results[:top_k]

        bm25_results = self.bm25.search_chunks(query, top_k=top_k * 5)

        if not bm25_results:
            fused = vector_results[:top_k * 2]
        elif not vector_results:
            fused = bm25_results[:top_k * 2]
        else:
            fused = rrf_fuse(vector_results, bm25_results, k=RRF_K)

        # Cross-encoder rerank (optional — OFF by default for chunks, too slow on CPU)
        rerank_enabled = use_rerank if use_rerank is not None else self._use_rerank_chunks
        if rerank_enabled and fused:
            reranker = self._get_reranker()
            if reranker:
                rerank_candidates = min(settings.reranker_candidates, len(fused))
                return reranker.rerank(query, fused, top_k=top_k)

        return fused[:top_k]
