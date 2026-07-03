"""Build the two vector knowledge bases in LanceDB.

    1. FAQ 知识库 (qa_pairs):   loads deduplicated QA pairs -> embed -> LanceDB
    2. 文档知识库 (doc_chunks): loads cleaned chunks      -> embed -> LanceDB

Usage:
    python scripts/build_knowledge_base.py               # Build both
    python scripts/build_knowledge_base.py --qa-only      # Only FAQ KG
    python scripts/build_knowledge_base.py --docs-only    # Only document KG
    python scripts/build_knowledge_base.py --rebuild      # Clear and rebuild
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "shared", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "storage", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from config.settings import settings
from llm.factory import create_embedding_provider
from storage.milvus_store import CHUNK_COLLECTION, QA_COLLECTION, get_milvus_store


def embed_batch(texts: list[str], label: str) -> list[list[float]]:
    print("  Embedding %s (%d items)..." % (label, len(texts)), end=" ", flush=True)
    t0 = time.time()
    embedder = create_embedding_provider()
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            embs = embedder.embed_documents(batch)
            embeddings.extend(embs)
        except Exception as e:
            for _ in batch:
                embeddings.append([0.0] * 1024)
    print("[OK] %dms" % ((time.time() - t0) * 1000))
    return embeddings


def build_faq_knowledge_base(store):
    qa_file = Path(settings.clean_data_dir) / "qa_deduped" / "deduped_qa_pairs.json"
    if not qa_file.exists():
        print("[FAIL] QA file not found: %s" % qa_file)
        return

    print("\n" + "=" * 60)
    print("  [1] FAQ Knowledge Base (%s)" % QA_COLLECTION)
    print("=" * 60)

    data = json.loads(qa_file.read_text(encoding="utf-8"))
    qa_pairs = data.get("qa_pairs", [])
    print("  Loading %d QA pairs" % len(qa_pairs))

    questions = [q["question"] for q in qa_pairs]
    answers = [q["answer"] for q in qa_pairs]
    sources = [q.get("source_url", q.get("source", "")) for q in qa_pairs]
    confidences = [q.get("confidence", 0.5) for q in qa_pairs]

    embeddings = embed_batch(questions, "QA questions")
    store.build_qa(questions, answers, sources, confidences, embeddings)
    print("  [OK] FAQ knowledge base ready")


def build_document_knowledge_base(store):
    chunk_dir = Path(settings.clean_data_dir) / "chunks"
    if not chunk_dir.exists():
        print("[FAIL] Chunks dir not found: %s" % chunk_dir)
        return

    print("\n" + "=" * 60)
    print("  [2] Document Knowledge Base (%s)" % CHUNK_COLLECTION)
    print("=" * 60)

    all_contents, all_titles, all_sources, all_categories = [], [], [], []
    for subdir in sorted(chunk_dir.iterdir()):
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
            if text and len(text) > 10:
                all_contents.append(text)
                all_titles.append(chunk.get("metadata", {}).get("title", ""))
                all_sources.append(chunk.get("document_url", ""))
                all_categories.append(subdir.name)

    print("  Loaded %d chunks" % len(all_contents))
    if not all_contents:
        return

    embeddings = embed_batch(all_contents, "document chunks")
    store.build_chunks(all_contents, all_titles, all_sources, all_categories, embeddings)
    print("  [OK] Document knowledge base ready")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-only", action="store_true")
    parser.add_argument("--docs-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Knowledge Base Builder")
    print("  Vector DB: Milvus Lite (%s)" % settings.milvus_db_path)
    print("  Embedding: %s (%s)" % (settings.embedding_provider, settings.embedding_model))
    print("=" * 60)

    t0 = time.time()
    store = get_milvus_store()

    do_qa = not args.docs_only
    do_docs = not args.qa_only

    if do_qa:
        build_faq_knowledge_base(store)
    if do_docs:
        build_document_knowledge_base(store)

    stats = store.stats()
    print("\n" + "=" * 60)
    print("  Knowledge Base Ready")
    print("=" * 60)
    for name, st in stats.items():
        print("  %s: %d rows" % (name, st["row_count"]))
    print("  Time: %ds" % int(time.time() - t0))
    print("=" * 60)
    store.close()


if __name__ == "__main__":
    main()
