"""Parse raw API JSON data into CleanDocument objects.

Handles each source type's specific data structure.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from shared.enums import DocumentSource, DocumentStatus
from shared.models import CleanDocument, QAPair, RawDocument

logger = logging.getLogger(__name__)


def load_raw_file(file_path: Path) -> dict:
    """Load a raw JSON file saved by the API fetcher."""
    return json.loads(file_path.read_text(encoding="utf-8"))


def _to_clean_text(html: str | None) -> str:
    """Rough HTML-to-text conversion. Replaced by cleaner module later."""
    if not html:
        return ""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&quot;", '"')
    return text


# ---------------------------------------------------------------------------
# Per-source parsers
# ---------------------------------------------------------------------------

def _parse_policy(item: dict) -> CleanDocument:
    content_html = item.get("content") or ""
    description = item.get("description") or ""
    title = item.get("title") or ""
    amount = item.get("amount") or ""
    apply_condition = item.get("applyCondition") or ""

    clean_content = _to_clean_text(content_html)
    # Combine description + cleaned content
    full_text = (description + "\n\n" + clean_content).strip()
    if amount:
        full_text = "补贴/金额: %s\n\n%s" % (amount, full_text)
    if apply_condition:
        full_text = full_text + "\n\n申请条件: %s" % apply_condition

    # Build source URL from policy ID when sourceUrl is empty
    policy_id = item.get("id", "")
    source_url = item.get("sourceUrl") or ""
    if not source_url and policy_id:
        source_url = "https://cqaip.cn/industry-news/" + policy_id
    if not source_url:
        source_url = "/api/policies"

    # Extract potential QA pair from the structured fields
    qa_pairs = []
    if title and (description or clean_content):
        # Combine description + clean content for richer answer
        parts = []
        if description:
            parts.append(description.strip())
        if clean_content:
            parts.append(clean_content[:800].strip())
        if amount:
            parts.append("补贴/金额: %s" % amount)
        answer = "\n\n".join(p for p in parts if p)
        qa_pairs.append(
            QAPair(
                question=title,
                answer=answer[:1000],
                source_url=source_url,
                confidence=0.9,
            )
        )
    # Use the "applyCondition" as supplementary info
    if apply_condition and title:
        qa_pairs.append(
            QAPair(
                question="%s - 申请条件" % title,
                answer=apply_condition[:600],
                source_url=source_url,
                confidence=0.8,
            )
        )
    # Amount standalone QA (if amount is a meaningful value)
    if amount and amount.strip() and len(amount.strip()) > 5 and title:
        qa_pairs.append(
            QAPair(
                question="%s - 补贴金额" % title,
                answer=amount.strip(),
                source_url=source_url,
                confidence=0.75,
            )
        )

    return CleanDocument(
        source_url=source_url,
        original_title=title,
        clean_title=title,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="政策",
        language="zh",
        extracted_qas=qa_pairs,
        source=DocumentSource.CQAIP_HTML,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "policies", "raw_id": item.get("id")},
    )


def _parse_industry_news(item: dict) -> CleanDocument:
    description = item.get("description") or ""
    content = item.get("content") or ""
    title = item.get("title") or ""

    clean_content = _to_clean_text(content)
    full_text = (description + "\n\n" + clean_content).strip()

    qa_pairs = []
    if title and description:
        qa_pairs.append(
            QAPair(
                question=title,
                answer=description,
                source_url=item.get("sourceUrl") or "/api/industry-news",
                confidence=0.7,
            )
        )

    return CleanDocument(
        source_url=item.get("sourceUrl") or "/api/industry-news",
        original_title=title,
        clean_title=title,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="行业资讯",
        language="zh",
        extracted_qas=qa_pairs,
        source=DocumentSource.CQAIP_HTML,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "industry_news", "raw_id": item.get("id")},
    )


def _parse_competition(item: dict) -> CleanDocument:
    description = item.get("description") or ""
    content = item.get("content") or ""
    title = item.get("title") or ""

    clean_content = _to_clean_text(content)
    full_text = (description + "\n\n" + clean_content).strip()

    source_url = "/api/competitions"

    qa_pairs = []
    if title and description:
        qa_pairs.append(
            QAPair(
                question=title,
                answer=description[:500],
                source_url=source_url,
                confidence=0.85,
            )
        )
    clean_content_stripped = clean_content.strip()
    if title and clean_content_stripped and len(clean_content_stripped) > 100:
        qa_pairs.append(
            QAPair(
                question="%s - 赛事详情" % title,
                answer=clean_content_stripped[:1000],
                source_url=source_url,
                confidence=0.7,
            )
        )

    return CleanDocument(
        source_url=source_url,
        original_title=title,
        clean_title=title,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="赛事活动",
        language="zh",
        source=DocumentSource.CQAIP_HTML,
        extracted_qas=qa_pairs,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "competitions", "raw_id": item.get("id")},
    )


def _parse_dataset(item: dict) -> CleanDocument:
    name = item.get("name") or ""
    description = item.get("description") or ""
    data_volume = item.get("dataVolume") or ""
    delivery_format = item.get("deliveryFormat") or ""
    category_name = ""
    if item.get("category"):
        category_name = item["category"].get("name", "")
    extra = item.get("extraFields") or {}

    parts = [description]
    if data_volume:
        parts.append("数据量: %s" % data_volume)
    if delivery_format:
        parts.append("交付格式: %s" % delivery_format)
    if isinstance(extra, dict):
        for k, v in extra.items():
            if v:
                parts.append("%s: %s" % (k, v))

    full_text = "\n".join(parts).strip()

    return CleanDocument(
        source_url="/api/datasets",
        original_title=name,
        clean_title=name,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="数据集",
        language="zh",
        source=DocumentSource.CQAIP_HTML,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "datasets", "raw_id": str(item.get("id", ""))},
    )


def _parse_marketplace(item: dict) -> CleanDocument:
    title = item.get("title") or item.get("name") or ""
    description = item.get("description") or ""
    full_text = (title + "\n" + description).strip() if title else description

    return CleanDocument(
        source_url="/api/marketplace",
        original_title=title,
        clean_title=title,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="供需市场",
        language="zh",
        source=DocumentSource.CQAIP_HTML,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "marketplace", "raw_id": str(item.get("id", ""))},
    )


def _parse_model(item: dict) -> CleanDocument:
    name = item.get("name") or ""
    provider = item.get("provider") or ""
    description = item.get("description") or ""
    tags = item.get("tags") or []

    parts = ["模型: %s" % name, "供应商: %s" % provider]
    if description:
        parts.append(description)
    if tags:
        parts.append("标签: %s" % ", ".join(tags) if isinstance(tags, list) else tags)
    full_text = "\n".join(parts).strip()

    return CleanDocument(
        source_url="/api/models",
        original_title=name,
        clean_title=name,
        clean_content=full_text,
        content_length_chars=len(full_text),
        category="模型",
        language="zh",
        source=DocumentSource.CQAIP_HTML,
        processed_time=datetime.utcnow(),
        cleaning_meta={"source_name": "models", "raw_id": str(item.get("id", ""))},
    )


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

PARSERS = {
    "policies": _parse_policy,
    "industry_news": _parse_industry_news,
    "competitions": _parse_competition,
    "datasets": _parse_dataset,
    "marketplace": _parse_marketplace,
    "models": _parse_model,
}


def parse_source(source_name: str, raw_file: Path) -> list[CleanDocument]:
    """Parse a raw data file into CleanDocument objects.

    Args:
        source_name: The data source name (e.g., "policies").
        raw_file: Path to the raw JSON file.

    Returns:
        List of CleanDocument objects.
    """
    parser = PARSERS.get(source_name)
    if parser is None:
        logger.warning("No parser for source: %s", source_name)
        return []

    data = load_raw_file(raw_file)
    items = data.get("items", [])
    logger.info("Parsing %s: %d items", source_name, len(items))

    documents = []
    for item in items:
        try:
            doc = parser(item)
            documents.append(doc)
        except Exception as e:
            logger.error("Failed to parse item in %s: %s", source_name, e)

    logger.info("  -> %d CleanDocuments", len(documents))
    return documents
