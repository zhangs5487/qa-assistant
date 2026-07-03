"""Deep crawler: extract nav QA pairs + crawl sub-page content APIs.

Handles two major data gaps:
    1. Navigation queries ("How do I find X?")
    2. Sub-page content (/docs, /user-manual, policy details, etc.)
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


def extract_nav_qa_pairs() -> list[dict]:
    """Parse nav_menus from site-config into navigation QA pairs.

    Returns list of dicts with question/answer about site navigation.
    """
    site_dir = Path(settings.raw_data_dir) / "site_config"
    json_files = list(site_dir.glob("*.json"))
    if not json_files:
        logger.warning("No site_config data found")
        return []

    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    items = data.get("items", [data])
    if not isinstance(items, list):
        items = [items]

    qa_pairs = []

    for item in items:
        nav_raw = item.get("nav_menus", "")
        if not nav_raw or not isinstance(nav_raw, str):
            continue
        try:
            menus = json.loads(nav_raw)
        except json.JSONDecodeError:
            continue

        # Per-menu QA: "How do I find X?"
        for m in menus:
            label = m.get("label", "")
            href = m.get("href", "")
            children = m.get("children", [])
            visible = m.get("visible", True)

            if not label or not visible:
                continue

            if children:
                # Generate QA: "How to find sub-items under X"
                sub_items = []
                for c in children:
                    if c.get("visible", True):
                        sub_items.append("- %s: %s" % (c.get("label", ""), c.get("href", "")))

                if sub_items:
                    q = "如何找到%s下的子功能？" % label
                    a = "%s包含以下子功能：\n%s" % (label, "\n".join(sub_items))
                    qa_pairs.append({
                        "question": q,
                        "answer": a,
                        "source_url": "/api/site-config",
                        "confidence": 1.0,
                    })

                # Generate individual QA for each child
                for c in children:
                    if not c.get("visible", True):
                        continue
                    clabel = c.get("label", "")
                    chref = c.get("href", "")
                    if clabel:
                        q = "%s在哪里？" % clabel
                        a = "%s位于【%s > %s】，访问路径：%s" % (
                            clabel, label, clabel,
                            "https://cqaip.cn%s" % chref if chref.startswith("/") else chref,
                        )
                        qa_pairs.append({
                            "question": q,
                            "answer": a,
                            "source_url": "/api/site-config",
                            "confidence": 1.0,
                        })

                        # More natural phrasings
                        qa_pairs.append({
                            "question": "怎么进入%s？" % clabel,
                            "answer": "进入方式：在顶部导航栏找到【%s】，点击【%s】即可访问。" % (label, clabel),
                            "source_url": "/api/site-config",
                            "confidence": 1.0,
                        })
            else:
                # Top-level page without children
                if label:
                    url = "https://cqaip.cn%s" % href if href.startswith("/") else href
                    qa_pairs.append({
                        "question": "%s在哪里？" % label,
                        "answer": "%s位于顶部导航栏，直接点击【%s】即可访问。\n链接：%s" % (label, label, url),
                        "source_url": "/api/site-config",
                        "confidence": 1.0,
                    })
                    qa_pairs.append({
                        "question": "怎么找到%s？" % label,
                        "answer": "在网站顶部导航栏找到【%s】，点击进入。\n页面地址：%s" % (label, url),
                        "source_url": "/api/site-config",
                        "confidence": 1.0,
                    })

        # Site-wide overview QA
        all_pages = []
        for m in menus:
            if m.get("visible", True):
                label = m.get("label", "")
                href = m.get("href", "")
                if label and href:
                    all_pages.append("- %s: %s" % (label, href))
                for c in m.get("children", []):
                    if c.get("visible", True):
                        all_pages.append("  - %s: %s" % (c.get("label", ""), c.get("href", "")))

        if all_pages:
            qa_pairs.append({
                "question": "网站有哪些功能模块？",
                "answer": "重庆市人工智能公共服务平台主要功能模块：\n%s" % "\n".join(all_pages),
                "source_url": "/api/site-config",
                "confidence": 1.0,
            })
            qa_pairs.append({
                "question": "平台提供哪些服务？",
                "answer": "平台提供以下服务：\n%s" % "\n".join(all_pages),
                "source_url": "/api/site-config",
                "confidence": 1.0,
            })

    logger.info("Generated %d navigation QA pairs", len(qa_pairs))
    return qa_pairs


# ---------------------------------------------------------------------------
# Sub-page content crawler
# ---------------------------------------------------------------------------

SUBPAGE_ENDPOINTS = [
    # Policy detail pages
    "/api/policies/{id}",
    # Try common content API patterns
    "/api/v1/page?slug=docs",
    "/api/v1/page/docs",
    "/api/public/page/docs",
    # Site pages
    "/api/v1/page/about",
    "/api/v1/page/user-manual",
    "/api/v1/page/service-agreement",
    "/api/v1/page/privacy-policy",
]


def crawl_policy_details() -> list[dict]:
    """Fetch individual policy detail pages for richer content.

    We already have policy IDs from /api/policies. Now fetch each one's detail.
    """
    base = settings.api_base_url.rstrip("/")
    cookies = {}
    if settings.casdoor_session_id:
        cookies["casdoor_session_id"] = settings.casdoor_session_id

    client = httpx.Client(
        headers={"Accept": "application/json"},
        cookies=cookies,
        timeout=30,
        follow_redirects=True,
    )

    results = []

    # First get policy IDs
    try:
        resp = client.get("%s/api/policies" % base)
        policy_list = resp.json().get("data", [])
    except Exception as e:
        logging.error("Failed to get policy list: %s", e)
        client.close()
        return results

    for p in policy_list:
        pid = p.get("id", "")
        if not pid:
            continue
        try:
            resp = client.get("%s/api/policies/%s" % (base, pid))
            if resp.status_code == 200:
                detail = resp.json()
                results.append({
                    "url": "/api/policies/%s" % pid,
                    "title": p.get("title", ""),
                    "content": detail.get("data", {}).get("content", "") if detail.get("data") else detail,
                    "source": "/api/policies",
                })
        except Exception as e:
            logging.warning("Failed to fetch policy detail %s: %s", pid, e)

    client.close()
    logger.info("Crawled %d policy detail pages", len(results))
    return results


def crawl_static_pages() -> list[dict]:
    """Try to fetch content from static-like pages.

    Some Next.js pages serve data through RSC or API endpoints.
    Try common patterns.
    """
    base = settings.api_base_url.rstrip("/")
    client = httpx.Client(
        headers={"Accept": "application/json"},
        timeout=30,
        follow_redirects=True,
    )

    results = []

    # Try known page slugs
    slugs = ["docs", "user-manual", "about", "service-agreement", "privacy-policy", "faq"]
    for slug in slugs:
        paths = [
            "/api/v1/page/%s" % slug,
            "/api/public/page/%s" % slug,
            "/api/page/%s" % slug,
            "/api/v1/pages/%s" % slug,
            "/api/cms/%s" % slug,
        ]
        for path in paths:
            try:
                resp = client.get("%s%s" % (base, path))
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("data", data)
                    if isinstance(content, dict) and (content.get("content") or content.get("title")):
                        results.append({
                            "url": path,
                            "title": content.get("title", slug),
                            "content": json.dumps(content, ensure_ascii=False),
                            "source": path,
                        })
                        break
            except Exception:
                continue

    client.close()
    logger.info("Crawled %d static pages", len(results))
    return results


def run_deep_crawl() -> dict:
    """Run deep crawl: navigation QA + policy details + static pages.

    Returns dict with all new data ready for embedding.
    """
    print("=" * 50)
    print("Deep Crawl: Navigation + Sub-page Content")
    print("=" * 50)

    # 1. Navigation QA pairs
    print("\n[1/3] Extracting navigation QA pairs from site config...")
    nav_qa = extract_nav_qa_pairs()

    # 2. Policy detail pages
    print("\n[2/3] Crawling policy detail pages...")
    policy_details = crawl_policy_details()

    # 3. Static pages
    print("\n[3/3] Crawling static pages...")
    static_pages = crawl_static_pages()

    # Save results
    out_dir = Path(settings.raw_data_dir) / "deep_crawl"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert policy details to QA pairs where possible
    content_qa = []
    for pd in policy_details:
        title = pd.get("title", "")
        content = str(pd.get("content", ""))
        if title and content:
            content_qa.append({
                "question": title,
                "answer": content[:1000] if len(content) > 1000 else content,
                "source_url": pd.get("url", "/api/policies"),
                "confidence": 0.9,
            })

    # Convert static pages to QA pairs
    for sp in static_pages:
        title = sp.get("title", "")
        content = str(sp.get("content", ""))
        if content and len(content) > 20:
            content_qa.append({
                "question": "%s的内容是什么？" % title,
                "answer": content[:1000] if len(content) > 1000 else content,
                "source_url": sp.get("url", ""),
                "confidence": 0.9,
            })

    all_new = {
        "timestamp": timestamp,
        "nav_qa_pairs": nav_qa,
        "policy_details": policy_details,
        "static_pages": static_pages,
        "content_qa_pairs": content_qa,
        "total_nav_qa": len(nav_qa),
        "total_policy_details": len(policy_details),
        "total_static_pages": len(static_pages),
        "total_content_qa": len(content_qa),
        "total_new_qa": len(nav_qa) + len(content_qa),
    }

    out_file = out_dir / ("%s.json" % timestamp)
    out_file.write_text(json.dumps(all_new, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved deep crawl results to: %s" % out_file)
    print("  Nav QA pairs:     %d" % len(nav_qa))
    print("  Policy details:   %d" % len(policy_details))
    print("  Static pages:     %d" % len(static_pages))
    print("  Content QA pairs: %d" % len(content_qa))
    print("  Total new QA:     %d" % all_new["total_new_qa"])

    return all_new
