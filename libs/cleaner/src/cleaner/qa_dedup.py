"""QA pair deduplication using embedding similarity.

After LLM augmentation, many variants may be near-duplicates.
This module uses vector embeddings to identify and merge similar questions.
"""

import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from config.settings import settings
from llm.base import ChatConfig, ChatMessage
from llm.factory import create_embedding_provider
from shared.models import QAPair

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _merge_answers(existing: QAPair, candidate: QAPair) -> str:
    """Merge two answers, preferring the longer/more detailed one."""
    if len(candidate.answer) > len(existing.answer):
        return candidate.answer
    return existing.answer


def deduplicate_qa_pairs(
    qa_pairs: Sequence[QAPair],
    threshold: float = 0.92,
    batch_size: int = 20,
) -> list[QAPair]:
    """Deduplicate QA pairs by question embedding similarity.

    Groups near-duplicate questions together, keeping the most natural
    variant (preferring the one with highest confidence, then shortest).

    Args:
        qa_pairs: QA pairs to deduplicate (original + augmented).
        threshold: Cosine similarity threshold for considering duplicates.
            Higher = stricter, fewer merges. 0.92 is a good default for
            Chinese FAQ paraphrases.
        batch_size: Number of embeddings to process per API call.

    Returns:
        Deduplicated list of QA pairs.
    """
    if not qa_pairs:
        return []

    logger.info("Deduplicating %d QA pairs (threshold=%.2f)...", len(qa_pairs), threshold)

    try:
        embedder = create_embedding_provider()
    except Exception as e:
        logger.error("Failed to create embedding provider: %s", e)
        logger.warning("Returning pairs without deduplication")
        return list(qa_pairs)

    # Get all question texts
    questions = [q.question.strip() for q in qa_pairs]

    # Embed in batches
    all_embeddings: list[list[float]] = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i : i + batch_size]
        try:
            embeddings = embedder.embed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug(
                "  Embedded [%d/%d]: %d questions",
                i + len(batch),
                len(questions),
                len(batch),
            )
        except Exception as e:
            logger.warning("Embedding batch failed [%d-%d]: %s", i, i + batch_size, e)
            # Add zero vectors for failed items
            for _ in batch:
                all_embeddings.append([0.0] * settings.embedding_dimensions)

    if not all_embeddings:
        return list(qa_pairs)

    # Greedy clustering: for each pair, merge if above threshold
    keep_indices: list[int] = []
    merged_indices: set[int] = set()

    for i in range(len(qa_pairs)):
        if i in merged_indices:
            continue
        keep_indices.append(i)

        # Check all subsequent pairs
        for j in range(i + 1, len(qa_pairs)):
            if j in merged_indices:
                continue
            if all_embeddings[i] and all_embeddings[j]:
                sim = _cosine_similarity(all_embeddings[i], all_embeddings[j])
                if sim >= threshold:
                    merged_indices.add(j)
                    # Merge answers (keep the better one)
                    qa_pairs[i].answer = _merge_answers(qa_pairs[i], qa_pairs[j])

    deduped = [qa_pairs[i] for i in keep_indices]
    removed = len(qa_pairs) - len(deduped)
    logger.info(
        "Dedup complete: %d -> %d (removed %d duplicates)",
        len(qa_pairs),
        len(deduped),
        removed,
    )

    return deduped


def run_dedup(
    input_path: str | Path = "",
    output_path: str | Path = "",
    threshold: float = 0.92,
) -> Path:
    """End-to-end: load augmented QA pairs -> dedup -> save.

    Args:
        input_path: Path to augmented QA pairs JSON.
            Defaults to data/clean/qa_augmented/augmented_qa_pairs.json.
        output_path: Where to save deduplicated QA pairs.
            Defaults to data/clean/qa_deduped/deduped_qa_pairs.json.
        threshold: Cosine similarity threshold.

    Returns:
        Path to saved deduplicated QA file.
    """
    in_path = Path(
        input_path
        or settings.clean_data_dir / "qa_augmented" / "augmented_qa_pairs.json"
    )
    if not in_path.exists():
        raise FileNotFoundError("Augmented QA file not found: %s" % in_path)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    qa_pairs = [QAPair(**item) for item in data.get("qa_pairs", [])]
    logger.info("Loaded %d QA pairs from %s", len(qa_pairs), in_path)

    deduped = deduplicate_qa_pairs(qa_pairs, threshold=threshold)

    out_dir = Path(output_path or str(Path(settings.clean_data_dir) / "qa_deduped"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "deduped_qa_pairs.json"

    payload = {
        "source": "qa_dedup",
        "threshold": threshold,
        "input_count": len(qa_pairs),
        "output_count": len(deduped),
        "qa_pairs": [q.model_dump(mode="json") for q in deduped],
    }

    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Saved %d deduplicated QA pairs to %s", len(deduped), out_file)

    return out_file
