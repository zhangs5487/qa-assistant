"""Milvus vector database (pymilvus 3.0 + milvus-lite 3.0).

Dev:  Milvus Lite embedded (local file).
Prod: http://milvus:19530 (standalone server).

Collections:
    qa_pairs    — FAQ knowledge base
    doc_chunks  — Document knowledge base
"""

import logging
import os
import random

from pymilvus import MilvusClient

from config.settings import settings

logger = logging.getLogger(__name__)

DIM = settings.embedding_dimensions
QA_COLLECTION = settings.milvus_qa_collection
CHUNK_COLLECTION = settings.milvus_chunk_collection


def _make_int_id() -> int:
    return random.randint(10**12, 10**15 - 1)


class MilvusStore:
    """Milvus vector store (pymilvus 3.0)."""

    def __init__(self, db_path: str = ""):
        uri = db_path or settings.milvus_db_path
        if uri.startswith("http"):
            self.client = MilvusClient(uri=uri)
        else:
            abs_path = os.path.abspath(uri)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            self.client = MilvusClient(abs_path)
        logger.info("Milvus connected: %s", uri)
        # Pre-load collections into memory for fast search
        for name in [QA_COLLECTION, CHUNK_COLLECTION]:
            try:
                if self.client.has_collection(name):
                    self.client.load_collection(name)
            except Exception:
                pass

    def _drop(self, name: str):
        try:
            if self.client.has_collection(name):
                self.client.drop_collection(name)
        except Exception as e:
            logger.debug("Could not drop %s: %s", name, e)

    def _create(self, name: str):
        self.client.create_collection(
            collection_name=name, dimension=DIM, auto_id=False
        )

    def _ensure_index(self, name: str):
        try:
            self.client.create_index(
                collection_name=name,
                index_params=self.client.prepare_index_params()
                    .add_index(field_name="vector", index_type="HNSW",
                               metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            )
        except Exception:
            pass
        self.client.load_collection(name)

    def build_qa(self, questions, answers, sources, confidences, vectors):
        self._drop(QA_COLLECTION)
        self._create(QA_COLLECTION)
        data = []
        for i in range(len(questions)):
            data.append({
                "id": _make_int_id(),
                "question": questions[i][:1024],
                "answer": answers[i][:4096],
                "source": sources[i][:512] if i < len(sources) else "",
                "confidence": float(confidences[i]) if i < len(confidences) else 0.5,
                "vector": vectors[i],
            })
        self.client.insert(collection_name=QA_COLLECTION, data=data)
        self._ensure_index(QA_COLLECTION)
        logger.info("QA built: %d rows", len(data))
        return len(data)

    def build_chunks(self, contents, titles, sources, categories, vectors):
        self._drop(CHUNK_COLLECTION)
        self._create(CHUNK_COLLECTION)
        data = []
        for i in range(len(contents)):
            data.append({
                "id": _make_int_id(),
                "content": contents[i][:8192],
                "title": titles[i][:512] if i < len(titles) else "",
                "source": sources[i][:512] if i < len(sources) else "",
                "category": categories[i][:128] if i < len(categories) else "",
                "vector": vectors[i],
            })
        self.client.insert(collection_name=CHUNK_COLLECTION, data=data)
        self._ensure_index(CHUNK_COLLECTION)
        logger.info("Chunks built: %d rows", len(data))
        return len(data)

    def search_qa(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if not self.client.has_collection(QA_COLLECTION):
            return []
        r = self.client.search(
            collection_name=QA_COLLECTION, data=[query_vector], limit=top_k,
            output_fields=["question", "answer", "source", "confidence"],
        )
        if not r:
            return []
        return [
            {
                "question": x["entity"].get("question", ""),
                "answer": x["entity"].get("answer", ""),
                "source": x["entity"].get("source", ""),
                "confidence": x["entity"].get("confidence", 0.5),
                "similarity": 1.0 - x.get("distance", 1.0),
            }
            for x in r[0]
        ]

    def search_chunks(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        if not self.client.has_collection(CHUNK_COLLECTION):
            return []
        r = self.client.search(
            collection_name=CHUNK_COLLECTION, data=[query_vector], limit=top_k,
            output_fields=["content", "title", "source", "category"],
        )
        if not r:
            return []
        return [
            {
                "content": x["entity"].get("content", ""),
                "title": x["entity"].get("title", ""),
                "source": x["entity"].get("source", ""),
                "category": x["entity"].get("category", ""),
                "similarity": 1.0 - x.get("distance", 1.0),
            }
            for x in r[0]
        ]

    def stats(self) -> dict:
        s = {}
        for name in [QA_COLLECTION, CHUNK_COLLECTION]:
            if self.client.has_collection(name):
                try:
                    s[name] = {"row_count": self.client.get_collection_stats(name)["row_count"]}
                except Exception:
                    s[name] = {"row_count": "unknown"}
        return s

    def close(self):
        if self.client:
            self.client.close()
        global _store
        _store = None


_store: "MilvusStore | None" = None


def get_milvus_store() -> "MilvusStore":
    global _store
    if _store is None:
        _store = MilvusStore()
    return _store
