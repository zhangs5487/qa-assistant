"""End-to-end QA processing: collect -> augment -> dedup -> save.

Usage:
    from cleaner.qa_processor import run_full_qa_pipeline
    run_full_qa_pipeline(num_variants=5, dedup_threshold=0.92)
"""

import logging
from pathlib import Path

from config.settings import settings

from .qa_augmenter import HARDCODED_FAQ, run_augmentation
from .qa_dedup import run_dedup

logger = logging.getLogger(__name__)


def run_full_qa_pipeline(
    num_variants: int = 5,
    dedup_threshold: float = 0.92,
) -> Path:
    """Full QA processing pipeline.

    Steps:
        1. Load existing QA pairs from cleaned documents
        2. Add hardcoded FAQ items
        3. Augment with LLM paraphrasing
        4. Deduplicate by embedding similarity
        5. Save final QA set

    Args:
        num_variants: Number of paraphrase variants per original QA pair.
        dedup_threshold: Cosine similarity threshold for dedup.

    Returns:
        Path to the final deduplicated QA pairs file.
    """
    print("=" * 60)
    print("QA Processing Pipeline")
    print("=" * 60)

    # Step 1 & 2 & 3: Load + augment
    print("\n[1/3] Loading and augmenting QA pairs...")
    aug_file = run_augmentation(num_variants=num_variants)
    print("  Augmented QA pairs saved to: %s" % aug_file)

    # Step 4 & 5: Dedup + save
    print("\n[2/3] Deduplicating...")
    dedup_file = run_dedup(
        input_path=aug_file,
        threshold=dedup_threshold,
    )
    print("  Deduplicated QA pairs saved to: %s" % dedup_file)

    # Result summary
    from .qa_augmenter import load_qa_pairs_from_documents

    originals = load_qa_pairs_from_documents()
    originals_with_faq = len(originals) + len([h for h in HARDCODED_FAQ
                                                if not any(q.question == h.question for q in originals)])

    import json
    final_data = json.loads(Path(dedup_file).read_text(encoding="utf-8"))
    final_count = final_data["output_count"]

    print("\n[3/3] Summary")
    print("  Original QA pairs:             %d" % originals_with_faq)
    print("  After augmentation (x%d):       %d" % (num_variants, final_data["input_count"]))
    print("  After dedup (threshold=%.2f):  %d" % (dedup_threshold, final_count))
    print("  Reduction:                     -%d (%.1f%%)" % (
        final_data["input_count"] - final_count,
        (1 - final_count / final_data["input_count"]) * 100 if final_data["input_count"] else 0,
    ))
    print("=" * 60)

    return Path(dedup_file)
