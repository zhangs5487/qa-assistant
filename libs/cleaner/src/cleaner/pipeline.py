"""Data cleaning pipeline: parse -> save clean -> chunk -> save chunks.

Orchestrates the processing of raw API data into clean, chunked documents
ready for embedding and storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

from config.settings import settings
from shared.models import Chunk, CleanDocument

from .chunker import DocumentChunker
from .parser import PARSERS, load_raw_file

logger = logging.getLogger(__name__)


def _save_clean(documents: list[CleanDocument], output_dir: Path) -> Path:
    """Save a list of CleanDocument objects to a JSON file.

    Returns the file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / ("%s.json" % timestamp)

    payload = {
        "processed_at": datetime.now().isoformat(),
        "count": len(documents),
        "documents": [d.model_dump(mode="json") for d in documents],
    }
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d clean documents to %s", len(documents), file_path)
    return file_path


def _save_chunks(chunks: list[Chunk], output_dir: Path) -> Path:
    """Save a list of Chunk objects to a JSON file.

    Returns the file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / ("%s.json" % timestamp)

    payload = {
        "processed_at": datetime.now().isoformat(),
        "count": len(chunks),
        "chunks": [c.model_dump(mode="json") for c in chunks],
    }
    file_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d chunks to %s", len(chunks), file_path)
    return file_path


def process_source(
    source_name: str,
    raw_file: Path,
    clean_dir: Path,
    chunk_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> tuple[list[CleanDocument], list[Chunk]]:
    """Process a single raw data source through the full pipeline.

    Steps:
        1. Parse raw items into CleanDocuments
        2. Save CleanDocuments to clean_dir
        3. Chunk documents
        4. Save chunks to chunk_dir

    Returns:
        (documents, chunks) tuple.
    """
    parser = PARSERS.get(source_name)
    if parser is None:
        logger.warning("No parser for source: %s", source_name)
        return [], []

    # 1. Load and parse
    raw_data = load_raw_file(raw_file)
    items = raw_data.get("items", [])
    logger.info("Processing %s: %d raw items", source_name, len(items))

    documents = []
    for item in items:
        try:
            doc = parser(item)
            documents.append(doc)
        except Exception as e:
            logger.error("Failed to parse item in %s: %s", source_name, e)

    if not documents:
        logger.warning("No documents produced for %s", source_name)
        return [], []

    # 2. Save clean documents
    _save_clean(documents, clean_dir)

    # 3. Chunk
    chunker = DocumentChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk_documents(documents)
    logger.info("  -> %d chunks produced", len(chunks))

    # 4. Save chunks
    _save_chunks(chunks, chunk_dir)

    return documents, chunks


def process_all(
    raw_base_dir: str | Path = "",
    clean_base_dir: str | Path = "",
    chunk_base_dir: str | Path = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_names: list[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Process all raw data sources through the pipeline.

    Args:
        raw_base_dir: Directory containing per-source raw data folders.
            Defaults to settings.raw_data_dir.
        clean_base_dir: Output directory for clean documents.
            Defaults to settings.clean_data_dir / "documents".
        chunk_base_dir: Output directory for chunks.
            Defaults to settings.clean_data_dir / "chunks".
        chunk_size: Token count per chunk.
        chunk_overlap: Token overlap between chunks.
        source_names: If set, only process these sources.

    Returns:
        Dict mapping source name -> (document_count, chunk_count).
    """
    raw_dir = Path(raw_base_dir or settings.raw_data_dir)
    clean_dir = Path(clean_base_dir or settings.clean_data_dir) / "documents"
    chunk_dir = Path(chunk_base_dir or settings.clean_data_dir) / "chunks"

    results = {}

    # Find the latest raw file for each source
    for source_dir in sorted(raw_dir.iterdir()):
        if not source_dir.is_dir():
            continue
        name = source_dir.name
        if source_names and name not in source_names:
            continue

        # Find latest .json file
        json_files = list(source_dir.glob("*.json"))
        if not json_files:
            logger.warning("No raw data for source: %s", name)
            continue

        latest = max(json_files, key=lambda p: p.stat().st_mtime)

        docs, chunks = process_source(
            source_name=name,
            raw_file=latest,
            clean_dir=clean_dir / name,
            chunk_dir=chunk_dir / name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        results[name] = (len(docs), len(chunks))

    return results
