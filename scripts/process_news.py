"""Process industry news: crawl full text -> chunk -> QA -> rebuild vector DB.

Usage:
    python scripts/process_news.py                    # Crawl + process + rebuild
    python scripts/process_news.py --crawl-only       # Only crawl full text
    python scripts/process_news.py --rebuild-only     # Only rebuild from existing enriched data
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "crawler", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "shared", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "llm", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "libs", "storage", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "config"))

from config.settings import settings
from shared.enums import DocumentSource, DocumentStatus
from shared.models import Chunk, CleanDocument, QAPair
from llm.factory import create_embedding_provider

logging.basicConfig(level=logging.WARNING)

import httpx


def _crawl_article(url: str) -> str:
    """Quick HTML-to-text fetch for a single article URL."""
    if not url:
        return ""
    import random
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
    try:
        resp = httpx.get(url, headers={"User-Agent": ua}, timeout=20, follow_redirects=True)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "lxml")
        for t in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
            t.decompose()
        for sel in [".content", ".article", ".main-text", ".news-content", "#content", "#article", ".rich_media_content"]:
            el = soup.select_one(sel)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text
        body = soup.find("body")
        if body:
            lines = [l.strip() for l in body.get_text(separator="\n", strip=True).split("\n") if len(l.strip()) > 20]
            return "\n".join(lines)
    except Exception:
        pass
    return ""


def crawl_news() -> list[dict]:
    """Crawl full text for all industry news items."""
    print("=" * 50)
    print("Crawling industry news full text...")
    print("=" * 50)

    raw_dir = Path(settings.raw_data_dir) / "industry_news"
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        print("No raw data found")
        return []

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    items = data.get("items", data.get("data", {}).get("items", []))

    enriched = []
    for i, item in enumerate(items):
        title = item.get("title", "")[:50]
        url = item.get("sourceUrl", "")
        desc = item.get("description", "")

        full = _crawl_article(url) if url else ""
        item["full_text"] = full or desc
        enriched.append(item)

        status = "OK" if full else "fallback"
        msg = "  [%d/%d] %s %s" % (i + 1, len(items), status, title[:40])
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
        time.sleep(0.3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = raw_dir / ("enriched_%s.json" % ts)
    out.write_text(json.dumps({"source": "news_crawler", "items": enriched}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved %d enriched items to %s" % (len(enriched), out))
    return enriched


def process_enriched() -> tuple[list[CleanDocument], list[Chunk]]:
    """Convert enriched news into CleanDocuments, chunks, and QA pairs.

    Returns:
        (documents, chunks)
    """
    print("\n" + "=" * 50)
    print("Processing enriched news -> documents + chunks + QA...")
    print("=" * 50)

    raw_dir = Path(settings.raw_data_dir) / "industry_news"
    json_files = sorted(raw_dir.glob("enriched_*.json"))
    if not json_files:
        print("No enriched data found. Run --crawl-only first.")
        return [], []

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    items = data.get("items", [])

    documents = []
    all_chunks = []
    all_qa_pairs = []

    for item in items:
        title = item.get("title") or ""
        desc = item.get("description") or ""
        full_text = item.get("full_text") or desc
        source_url = item.get("sourceUrl") or "/api/industry-news"
        category = item.get("category") or "行业资讯"

        # Combine description + full text
        clean_content = (desc + "\n\n" + full_text).strip() if desc and full_text else (full_text or desc)
        if not clean_content:
            continue

        # QA pair: title -> content
        qas = []
        if title and clean_content:
            qas.append(QAPair(question=title, answer=clean_content[:1000], source_url=source_url, confidence=0.8))

        doc = CleanDocument(
            source_url=source_url,
            original_title=title,
            clean_title=title,
            clean_content=clean_content,
            content_length_chars=len(clean_content),
            category=category,
            language="zh",
            extracted_qas=qas,
            source=DocumentSource.CQAIP_HTML,
            processed_time=datetime.utcnow(),
            cleaning_meta={"source_name": "industry_news_enriched"},
        )
        documents.append(doc)

        # Simple chunking (no LangChain needed for 34 articles)
        chunk_size = 500
        overlap = 50
        text = clean_content
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Chunk(
                    document_url=source_url,
                    chunk_index=idx,
                    content=chunk_text,
                    token_count=len(chunk_text),
                    overlap_with_prev=idx > 0,
                    metadata={"title": title, "category": category, "source": source_url},
                ))
                idx += 1
            start += chunk_size - overlap

        all_chunks.extend(chunks)

        safe_title = title.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        print("  [OK] %s -> %d chunks, %d QA" % (safe_title[:40], len(chunks), len(qas)))

    # Save documents
    doc_dir = Path(settings.clean_data_dir) / "documents" / "industry_news"
    doc_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_file = doc_dir / ("%s.json" % ts)
    doc_payload = {
        "processed_at": datetime.now().isoformat(),
        "count": len(documents),
        "documents": [d.model_dump(mode="json") for d in documents],
    }
    doc_file.write_text(json.dumps(doc_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save chunks
    chunk_dir = Path(settings.clean_data_dir) / "chunks" / "industry_news"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = chunk_dir / ("%s.json" % ts)
    chunk_payload = {
        "processed_at": datetime.now().isoformat(),
        "count": len(all_chunks),
        "chunks": [c.model_dump(mode="json") for c in all_chunks],
    }
    chunk_file.write_text(json.dumps(chunk_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved: %d docs, %d chunks" % (len(documents), len(all_chunks)))
    return documents, all_chunks


def rebuild_knowledge_bases():
    """Rebuild both Milvus collections from scratch."""
    print("\n" + "=" * 50)
    print("Rebuilding knowledge bases...")
    print("=" * 50)

    from storage.milvus_store import get_milvus_store, QA_COLLECTION, CHUNK_COLLECTION
    from cleaner.chunker import DocumentChunker

    store = get_milvus_store()
    embedder = create_embedding_provider()

    # Collect all docs and chunks
    all_docs = []
    doc_base = Path(settings.clean_data_dir) / "documents"
    if doc_base.exists():
        for subdir in sorted(doc_base.iterdir()):
            json_files = sorted(subdir.glob("*.json"))
            if not json_files:
                continue
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                continue
            for d in data.get("documents", []):
                all_docs.append(CleanDocument(**d))

    all_chunks = []
    chunk_base = Path(settings.clean_data_dir) / "chunks"
    if chunk_base.exists():
        for subdir in sorted(chunk_base.iterdir()):
            json_files = sorted(subdir.glob("*.json"))
            if not json_files:
                continue
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            try:
                data = json.loads(latest.read_text(encoding="utf-8"))
            except Exception:
                continue
            for c in data.get("chunks", []):
                all_chunks.append(Chunk(**c))

    # QA pairs from documents
    qa_pairs = []
    for doc in all_docs:
        for qa in doc.extracted_qas:
            if qa.question and qa.answer:
                qa_pairs.append(qa)

    # Also load the augmented+deduplicated QA pairs
    qa_dedup_file = Path(settings.clean_data_dir) / "qa_deduped" / "deduped_qa_pairs.json"
    if qa_dedup_file.exists():
        import json
        qa_data = json.loads(qa_dedup_file.read_text(encoding="utf-8"))
        for item in qa_data.get("qa_pairs", []):
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                qa_pairs.append(QAPair(
                    question=q, answer=a,
                    source_url=item.get("source_url", ""),
                    confidence=item.get("confidence", 0.5),
                ))

    # Dedup by question
    seen = set()
    unique_qa = []
    for q in qa_pairs:
        k = q.question.strip()
        if k not in seen:
            seen.add(k)
            unique_qa.append(q)

    print("\n  Total: %d docs, %d chunks, %d QA pairs" % (len(all_docs), len(all_chunks), len(unique_qa)))

    # Embed QA
    print("  Embedding QA pairs...", end=" ", flush=True)
    t0 = time.time()
    qa_vecs = []
    batch_size = 20
    for i in range(0, len(unique_qa), batch_size):
        batch = [q.question for q in unique_qa[i:i+batch_size]]
        try:
            vecs = embedder.embed_documents(batch)
            qa_vecs.extend(vecs)
        except Exception:
            for _ in batch:
                qa_vecs.append([0.0] * 1024)
    print("%ds" % (time.time() - t0))

    # Build QA collection
    store.build_qa(
        questions=[q.question for q in unique_qa],
        answers=[q.answer for q in unique_qa],
        sources=[q.source_url or "" for q in unique_qa],
        confidences=[q.confidence or 0.5 for q in unique_qa],
        vectors=qa_vecs,
    )

    # Embed chunks
    print("  Embedding chunks...", end=" ", flush=True)
    t0 = time.time()
    chunk_vecs = []
    for i in range(0, len(all_chunks), batch_size):
        batch = [c.content for c in all_chunks[i:i+batch_size]]
        try:
            vecs = embedder.embed_documents(batch)
            chunk_vecs.extend(vecs)
        except Exception:
            for _ in batch:
                chunk_vecs.append([0.0] * 1024)
    print("%ds" % (time.time() - t0))

    # Build chunk collection
    store.build_chunks(
        contents=[c.content for c in all_chunks],
        titles=[c.metadata.get("title", "") for c in all_chunks],
        sources=[c.document_url for c in all_chunks],
        categories=[c.metadata.get("category", "") for c in all_chunks],
        vectors=chunk_vecs,
    )

    store.close()
    print("\n[OK] Knowledge bases rebuilt")
    print("  qa_pairs:   %d" % len(unique_qa))
    print("  doc_chunks: %d" % len(all_chunks))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--crawl-only", action="store_true", help="Only crawl news full text")
    p.add_argument("--rebuild-only", action="store_true", help="Only rebuild from existing data")
    args = p.parse_args()

    do_crawl = not args.rebuild_only
    do_rebuild = not args.crawl_only

    if do_crawl:
        items = crawl_news()
        if items:
            process_enriched()

    if do_rebuild:
        rebuild_knowledge_bases()

    print("\nDone!")


if __name__ == "__main__":
    main()
