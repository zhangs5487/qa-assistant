"""Crawl full text content from industry news source URLs.

The /api/industry-news endpoint returns 34 articles with empty 'content'
fields. Each article has a 'sourceUrl' pointing to the original news page.
This module fetches those pages and extracts the article text.
"""

import json
import logging
import re
import time
from pathlib import Path
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/604.1",
]


def _extract_text(html: str, url: str) -> str:
    """Extract main article text from HTML using multiple strategies."""
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: article tag
    article = soup.find("article")
    if article:
        text = article.get_text(separator="\n", strip=True)
        if len(text) > 200:
            return text

    # Strategy 2: common content selectors
    for selector in [
        ".content", ".article", ".main-text", ".news-content",
        "#content", "#article", ".post-content", ".entry-content",
        ".rich_media_content", ".rich_media_area_primary",
    ]:
        el = soup.select_one(selector)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                return text

    # Strategy 3: body text (remove nav, footer, header)
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    body = soup.find("body")
    if body:
        text = body.get_text(separator="\n", strip=True)
        # Filter short lines
        lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
        text = "\n".join(lines)
        if len(text) > 200:
            return text

    return ""


def crawl_article(url: str) -> str:
    """Fetch a single article URL and extract its text content.

    Args:
        url: Source URL of the news article.

    Returns:
        Extracted article text, or empty string if failed.
    """
    if not url:
        return ""

    import random
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    try:
        resp = httpx.get(url, headers=headers, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        text = _extract_text(resp.text, url)
        if text:
            logger.debug("  -> %d chars from %s", len(text), url)
        else:
            logger.warning("  -> empty extraction from %s", url)
        return text
    except httpx.HTTPStatusError as e:
        logger.warning("  -> HTTP %d for %s", e.response.status_code, url)
    except httpx.RequestError as e:
        logger.warning("  -> request failed: %s", e)
    except Exception as e:
        logger.warning("  -> error: %s", e)
    return ""


def crawl_all_news(output_file: str = "") -> list[dict]:
    """Fetch all 34 industry news items and crawl their full text.

    Args:
        output_file: Path to save the enriched news data.

    Returns:
        List of enriched news items with 'full_text' field added.
    """
    # Load raw industry news data
    raw_dir = Path(settings.raw_data_dir) / "industry_news"
    json_files = sorted(raw_dir.glob("*.json"))
    if not json_files:
        logger.error("No industry news raw data found")
        return []
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    items = data.get("items", [])
    if not items:
        # Check if wrapped in data key
        if "data" in data and isinstance(data["data"], dict):
            items = data["data"].get("items", [])

    print("Industry news: %d items to crawl" % len(items))

    enriched = []
    success = 0
    for i, item in enumerate(items):
        title = item.get("title", "")[:60]
        url = item.get("sourceUrl", "")
        desc = item.get("description", "")

        article = crawl_article(url) if url else ""

        enriched_item = dict(item)
        enriched_item["full_text"] = article or desc  # fallback to description
        enriched.append(enriched_item)

        if article:
            success += 1

        print("  [%d/%d] %s ... %d chars" % (i + 1, len(items), title, len(article or desc)))

        # Polite delay
        if url and article:
            time.sleep(0.5)

    print("\nCrawled: %d/%d articles with full text" % (success, len(items)))

    # Save enriched data
    if output_file:
        out_path = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(settings.raw_data_dir) / "industry_news" / ("enriched_%s.json" % timestamp)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "industry_news_crawler",
        "crawled_at": datetime.now().isoformat(),
        "total": len(enriched),
        "with_full_text": success,
        "items": enriched,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved enriched news to: %s" % out_path)

    return enriched
