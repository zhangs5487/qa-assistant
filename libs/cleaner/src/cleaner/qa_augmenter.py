"""QA pair augmentation using LLM paraphrase generation.

Takes existing QA pairs from parsed documents and generates multiple
question variants using the configured ChatProvider (ZhipuAI GLM).
"""

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from config.settings import settings
from llm.base import ChatConfig, ChatMessage
from llm.factory import create_chat_provider
from shared.models import QAPair

logger = logging.getLogger(__name__)

# Fixed FAQ items found embedded in the cqaip.cn policies page JS code
HARDCODED_FAQ: list[QAPair] = [
    QAPair(
        question="什么样的企业可以申请入驻OPC社区？",
        answer="面向AI领域的初创企业、个体创业者、AI技术开发者和高校创业团队均可申请，无需特定营业额要求。",
        source_url="/api/policies",
        confidence=1.0,
    ),
    QAPair(
        question="政策补贴需要多久可以到账？",
        answer="一般情况下，审核通过后30个工作日内完成补贴发放，具体以政府部门公告为准。",
        source_url="/api/policies",
        confidence=1.0,
    ),
    QAPair(
        question="入驻后如何享受各项权益？",
        answer="完成企业认证后，可在平台上直接申请各类资源，社区主理人会全程辅助您完成权益兑换。",
        source_url="/api/policies",
        confidence=1.0,
    ),
    QAPair(
        question="算力和Token补贴可以叠加使用吗？",
        answer="各类补贴政策可以叠加享受，具体叠加规则请联系社区运营团队确认。",
        source_url="/api/policies",
        confidence=1.0,
    ),
]


AUGMENT_PROMPT_TEMPLATE = """你是一个专业的FAQ问答对改写助手。请对以下问答对进行改写，生成{num_variants}个不同的问法变体。

要求：
- 保持答案不变
- 只改写问题部分，生成不同的提问方式
- 覆盖以下风格：口语化问法、书面语问法、简短问法、详细问法、同义词替换
- 每个变体一行，不要编号
- 不要改变原意

原问题：{question}
原答案：{answer}

改写问题（每行一个）："""


def _parse_variants(text: str) -> list[str]:
    """Parse the LLM response into a list of question variants."""
    lines = text.strip().split("\n")
    variants = []
    for line in lines:
        line = line.strip()
        # Remove common prefixes like "- ", "1. ", "* "
        line = re.sub(r"^[\-\*\d]+\.\s*", "", line).strip()
        # Remove quotes
        line = line.strip("\"'「」")
        if line and len(line) > 4:
            variants.append(line)
    return variants


def load_qa_pairs_from_documents(clean_dir: str | Path = "") -> list[QAPair]:
    """Load all QA pairs from saved clean document JSON files.

    Args:
        clean_dir: Base clean data directory. Defaults to settings.clean_data_dir.

    Returns:
        Deduplicated list of QAPair objects.
    """
    base = Path(clean_dir or settings.clean_data_dir) / "documents"
    if not base.exists():
        logger.warning("Clean documents directory not found: %s", base)
        return []

    seen = set()
    qa_pairs = []

    for source_dir in sorted(base.iterdir()):
        if not source_dir.is_dir():
            continue
        json_files = list(source_dir.glob("*.json"))
        if not json_files:
            continue
        latest = max(json_files, key=lambda p: p.stat().st_mtime)

        try:
            data = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to load %s: %s", latest, e)
            continue

        for doc in data.get("documents", []):
            for qa_dict in doc.get("extracted_qas", []):
                qa = QAPair(**qa_dict)
                # Dedup by question text
                key = qa.question.strip()
                if key and key not in seen:
                    seen.add(key)
                    qa_pairs.append(qa)

    logger.info("Loaded %d QA pairs from clean documents", len(qa_pairs))
    return qa_pairs


def augment_qa_pairs(
    qa_pairs: Sequence[QAPair],
    num_variants: int = 5,
    batch_size: int = 5,
) -> list[QAPair]:
    """Use LLM to generate paraphrased question variants for each QA pair.

    Args:
        qa_pairs: Original QA pairs to augment.
        num_variants: Number of paraphrase variants per pair.
        batch_size: Number of QA pairs to process per LLM call.

    Returns:
        Combined list: original pairs + augmented variants.
    """
    if not qa_pairs:
        return []

    logger.info(
        "Augmenting %d QA pairs with %d variants each...",
        len(qa_pairs),
        num_variants,
    )

    try:
        chat = create_chat_provider()
    except Exception as e:
        logger.error("Failed to create chat provider: %s", e)
        logger.warning("Returning original pairs without augmentation")
        return list(qa_pairs)

    config = ChatConfig(temperature=0.8, max_tokens=1024)
    results = list(qa_pairs)  # Keep originals

    for i, qa in enumerate(qa_pairs):
        logger.debug(
            "Augmenting [%d/%d]: %s...",
            i + 1,
            len(qa_pairs),
            qa.question[:40],
        )

        prompt = AUGMENT_PROMPT_TEMPLATE.format(
            num_variants=num_variants,
            question=qa.question,
            answer=qa.answer,
        )

        try:
            response = chat.chat(
                messages=[ChatMessage(role="user", content=prompt)],
                config=config,
            )
            variants = _parse_variants(response)

            for vq in variants[:num_variants]:
                results.append(
                    QAPair(
                        question=vq,
                        answer=qa.answer,
                        source_url=qa.source_url,
                        source_doc_id=qa.source_doc_id,
                        confidence=0.7,  # LLM-generated, lower confidence
                    )
                )

            logger.debug("  -> %d variants generated", len(variants))

        except Exception as e:
            logger.warning("Failed to augment [%d/%d]: %s", i + 1, len(qa_pairs), e)

    logger.info("Augmentation complete: %d -> %d pairs", len(qa_pairs), len(results))
    return results


def load_deep_crawl_qa_pairs(raw_dir: str | Path = "") -> list[QAPair]:
    """Load QA pairs from deep crawl results (nav QA + content QA).

    The deep crawler generates nav QA pairs from site structure and
    content QA from sub-page crawling. These are high-confidence
    (confidence=1.0 or 0.9) and typically don't need augmentation.

    Args:
        raw_dir: Raw data directory. Defaults to settings.raw_data_dir.

    Returns:
        List of QAPair objects from deep crawl.
    """
    base = Path(raw_dir or settings.raw_data_dir) / "deep_crawl"
    if not base.exists():
        logger.info("No deep crawl data found at %s", base)
        return []

    json_files = sorted(base.glob("*.json"))
    if not json_files:
        return []

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load deep crawl data: %s", e)
        return []

    qa_pairs = []

    # Load nav QA pairs
    for item in data.get("nav_qa_pairs", []):
        if item.get("question") and item.get("answer"):
            qa_pairs.append(QAPair(
                question=item["question"],
                answer=item["answer"],
                source_url=item.get("source_url", ""),
                confidence=item.get("confidence", 1.0),
            ))

    # Load content QA pairs
    for item in data.get("content_qa_pairs", []):
        if item.get("question") and item.get("answer"):
            qa_pairs.append(QAPair(
                question=item["question"],
                answer=item["answer"],
                source_url=item.get("source_url", ""),
                confidence=item.get("confidence", 0.9),
            ))

    logger.info("Loaded %d QA pairs from deep crawl", len(qa_pairs))
    return qa_pairs


def run_augmentation(
    clean_dir: str | Path = "",
    output_dir: str | Path = "",
    num_variants: int = 5,
) -> Path:
    """End-to-end: load existing QA pairs -> augment -> save.

    Args:
        clean_dir: Directory with clean documents.
        output_dir: Where to save the augmented QA set.
        num_variants: Variants per QA pair.

    Returns:
        Path to the saved augmented QA file.
    """
    # Step 1: Load existing QA pairs from documents
    qa_pairs = load_qa_pairs_from_documents(clean_dir)

    # Step 2: Load deep crawl QA pairs (navigation + sub-page content)
    deep_qa = load_deep_crawl_qa_pairs()
    for dq in deep_qa:
        if not any(q.question == dq.question for q in qa_pairs):
            qa_pairs.append(dq)

    # Step 3: Generate platform knowledge QA pairs (nav + usage)
    from .platform_knowledge import generate_all_platform_knowledge
    nav_qa, platform_qa = generate_all_platform_knowledge()
    for pq in nav_qa + platform_qa:
        if not any(q.question.strip() == pq.question.strip() for q in qa_pairs):
            qa_pairs.append(pq)
    logger.info("Added %d platform knowledge QA pairs", len(nav_qa) + len(platform_qa))

    # Step 4: Add hardcoded FAQ items
    for hf in HARDCODED_FAQ:
        if not any(q.question == hf.question for q in qa_pairs):
            qa_pairs.append(hf)
            logger.info("Added hardcoded FAQ: %s", hf.question)

    logger.info("Total QA pairs before augmentation: %d", len(qa_pairs))

    # Step 3: Augment
    augmented = augment_qa_pairs(qa_pairs, num_variants=num_variants)

    # Step 4: Save
    out_dir = Path(output_dir or settings.clean_data_dir) / "qa_augmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "augmented_qa_pairs.json"

    payload = {
        "source": "qa_augmenter",
        "total_original": len(qa_pairs),
        "total_augmented": len(augmented),
        "qa_pairs": [q.model_dump(mode="json") for q in augmented],
    }

    out_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Saved %d QA pairs (original: %d, augmented: %d) to %s",
        len(augmented),
        len(qa_pairs),
        len(augmented) - len(qa_pairs),
        out_file,
    )

    return out_file
