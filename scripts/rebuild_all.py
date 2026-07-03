"""Complete rebuild: delete Milvus DB -> reconnect -> rebuild from scratch.

Usage:
    python scripts/rebuild_all.py
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "shared", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "storage", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from config.settings import settings
from llm.factory import create_embedding_provider
from shared.models import Chunk, CleanDocument, QAPair


def main():
    print("=" * 50)
    print("  Step 1: Load data from disk")
    print("=" * 50)

    # Load all documents
    all_docs = []
    doc_base = Path(settings.clean_data_dir) / "documents"
    for sd in sorted(doc_base.iterdir()):
        jfs = sorted(sd.glob("*.json"))
        if not jfs:
            continue
        latest = max(jfs, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        for d in data.get("documents", []):
            all_docs.append(CleanDocument(**d))
    print("  Documents: %d" % len(all_docs))

    # Load all chunks
    all_chunks = []
    chunk_base = Path(settings.clean_data_dir) / "chunks"
    for sd in sorted(chunk_base.iterdir()):
        jfs = sorted(sd.glob("*.json"))
        if not jfs:
            continue
        latest = max(jfs, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        for c in data.get("chunks", []):
            all_chunks.append(Chunk(**c))
    print("  Chunks: %d" % len(all_chunks))

    # Collect QA pairs
    qa_pairs = []
    for doc in all_docs:
        for qa in doc.extracted_qas:
            if qa.question and qa.answer:
                qa_pairs.append(QAPair(
                    question=qa.question, answer=qa.answer,
                    source_url=qa.source_url, confidence=qa.confidence,
                ))

    qa_file = Path(settings.clean_data_dir) / "qa_deduped" / "deduped_qa_pairs.json"
    if qa_file.exists():
        for item in json.loads(qa_file.read_text(encoding="utf-8")).get("qa_pairs", []):
            q, a = item.get("question", "").strip(), item.get("answer", "").strip()
            if q and a and not any(q == qa.question for qa in qa_pairs):
                qa_pairs.append(QAPair(
                    question=q, answer=a,
                    source_url=item.get("source_url", ""),
                    confidence=item.get("confidence", 0.5),
                ))

    seen = set()
    unique_qa = []
    for q in qa_pairs:
        k = q.question.strip()
        if k not in seen:
            seen.add(k)
            unique_qa.append(q)
    print("  QA pairs: %d" % len(unique_qa))

    # Step 2: Delete Milvus DB
    print("\n" + "=" * 50)
    print("  Step 2: Delete old Milvus DB")
    print("=" * 50)

    milvus_path = Path(settings.milvus_db_path).absolute()
    print("  Deleting: %s" % milvus_path)
    if milvus_path.exists():
        shutil.rmtree(milvus_path)
    milvus_path.mkdir(parents=True, exist_ok=True)
    print("  [OK] Deleted and recreated")

    # Step 3: Rebuild
    print("\n" + "=" * 50)
    print("  Step 3: Rebuild from scratch")
    print("=" * 50)

    from storage.milvus_store import get_milvus_store, QA_COLLECTION, CHUNK_COLLECTION

    store = get_milvus_store()
    embedder = create_embedding_provider()

    # Embed QA
    print("  Embedding %d QA pairs..." % len(unique_qa), end=" ", flush=True)
    t0 = time.time()
    qa_vecs = []
    for i in range(0, len(unique_qa), 20):
        try:
            qa_vecs.extend(embedder.embed_documents([q.question for q in unique_qa[i:i+20]]))
        except Exception:
            for _ in unique_qa[i:i+20]:
                qa_vecs.append([0.0] * settings.embedding_dimensions)
    print("%ds" % (time.time() - t0))

    store.build_qa(
        [q.question for q in unique_qa],
        [q.answer for q in unique_qa],
        [q.source_url or "" for q in unique_qa],
        [q.confidence or 0.5 for q in unique_qa],
        qa_vecs,
    )
    print("  [OK] qa_pairs: %d rows" % len(unique_qa))

    # Embed chunks
    print("  Embedding %d chunks..." % len(all_chunks), end=" ", flush=True)
    t0 = time.time()
    chunk_vecs = []
    for i in range(0, len(all_chunks), 20):
        try:
            chunk_vecs.extend(embedder.embed_documents([c.content for c in all_chunks[i:i+20]]))
        except Exception:
            for _ in all_chunks[i:i+20]:
                chunk_vecs.append([0.0] * settings.embedding_dimensions)
    print("%ds" % (time.time() - t0))

    store.build_chunks(
        [c.content for c in all_chunks],
        [c.metadata.get("title", "") for c in all_chunks],
        [c.document_url for c in all_chunks],
        [c.metadata.get("category", "") for c in all_chunks],
        chunk_vecs,
    )
    print("  [OK] doc_chunks: %d rows" % len(all_chunks))

    store.close()
    print("\n" + "=" * 50)
    print("  DONE")
    print("  qa_pairs:   %d" % len(unique_qa))
    print("  doc_chunks: %d" % len(all_chunks))
    print("=" * 50)


if __name__ == "__main__":
    main()
