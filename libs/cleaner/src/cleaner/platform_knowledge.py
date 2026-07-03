"""Platform Knowledge Generator.

Uses LLM to generate comprehensive platform usage QA pairs based on
the site configuration data, menu structure, and platform description.
This fills the gap between raw API data and user-facing Q&A.

Covers:
    - Navigation guidance ("where is X?")
    - Platform features ("what does X do?")
    - User journeys ("how do I use X?")
    - Registration & onboarding
"""

import json
import logging
from pathlib import Path
from typing import Sequence

from config.settings import settings
from llm.base import ChatConfig, ChatMessage
from llm.factory import create_chat_provider
from shared.models import QAPair

logger = logging.getLogger(__name__)


def _load_site_config() -> dict:
    """Load the latest site configuration from raw data."""
    config_dir = Path(settings.raw_data_dir) / "site_config"
    json_files = sorted(config_dir.glob("*.json"))
    if not json_files:
        logger.warning("No site_config found")
        return {}
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    data = json.loads(latest.read_text(encoding="utf-8"))
    items = data.get("items", [data])
    if isinstance(items, list):
        return items[0] if items else {}
    return items


def _extract_platform_description(config: dict) -> dict:
    """Extract key platform facts from site config."""
    nav_raw = config.get("nav_menus", "")
    menus = json.loads(nav_raw) if isinstance(nav_raw, str) else nav_raw

    # Collect all visible pages
    pages = []
    for m in menus:
        if m.get("visible", True):
            pages.append({
                "section": m.get("label", ""),
                "href": m.get("href", ""),
                "children": [
                    {"label": c.get("label", ""), "href": c.get("href", "")}
                    for c in m.get("children", [])
                    if c.get("visible", True)
                ],
            })

    return {
        "site_name": config.get("site_name", "重庆市人工智能公共服务平台"),
        "description": config.get("site_description", ""),
        "hero_title": config.get("hero_title", ""),
        "stats": {
            "users": config.get("stats_users", ""),
            "tools": config.get("stats_tools", ""),
            "solutions": config.get("stats_solutions", ""),
        },
        "pages": pages,
        "enable_register": config.get("enable_register", "true"),
        "enable_captcha": config.get("enable_captcha", "true"),
        "contact_phone": config.get("contact_phone", ""),
        "contact_email": config.get("contact_email", ""),
        "icp_number": config.get("icp_number", ""),
    }


PLATFORM_QA_GENERATION_PROMPT = """你是一个AI公共服务平台的客服专家。请根据以下平台信息，生成全面的FAQ问答对。

## 平台信息
平台名称: {site_name}
平台定位: {description}
支持的注册: {enable_register}

## 平台功能模块
{modules}

## 已知政策
{policies_summary}

## 统计数据
用户: {stats_users} | 工具: {stats_tools} | 解决方案: {stats_solutions}

---

请针对以下场景各生成3-5个问答对（共20个）：

1. **平台导航**: 用户想找到某个功能模块，问"XXX在哪里"、"怎么找XXX"
2. **功能咨询**: 用户想了解某个功能是做什么的
3. **使用流程**: 用户想知道如何完成特定任务（注册、申请、使用模型、购买算力等）
4. **入驻相关**: 用户想了解入驻条件、流程、权益
5. **常见问题**: 用户可能遇到的通用问题

格式要求（每行一个）：
Q: [问题]
A: [回答]

直接输出问答对，不要编号，不要太长（每个回答100字以内）。"""


def generate_platform_qa_pairs() -> list[QAPair]:
    """Use LLM to generate comprehensive platform usage QA pairs.

    Returns:
        List of QAPair objects covering platform navigation, features,
        user journeys, and common questions.
    """
    config = _load_site_config()
    if not config:
        return []

    info = _extract_platform_description(config)

    # Format modules
    modules_text = ""
    for p in info["pages"]:
        modules_text += "\n- %s (%s)" % (p["section"], p["href"])
        for c in p["children"]:
            modules_text += "\n    - %s (%s)" % (c["label"], c["href"])

    # Load policy titles as summary
    policies_dir = Path(settings.raw_data_dir) / "policies"
    policy_json_files = sorted(policies_dir.glob("*.json")) if policies_dir.exists() else []
    policy_titles = []
    if policy_json_files:
        latest_policies = max(policy_json_files, key=lambda p: p.stat().st_mtime)
        try:
            pdata = json.loads(latest_policies.read_text(encoding="utf-8"))
            for item in pdata.get("items", [])[:10]:
                policy_titles.append(item.get("title", ""))
        except Exception:
            pass

    prompt = PLATFORM_QA_GENERATION_PROMPT.format(
        site_name=info["site_name"],
        description=info["description"],
        enable_register=info["enable_register"],
        modules=modules_text,
        policies_summary="\n".join(policy_titles[:5]) if policy_titles else "暂无",
        stats_users=info["stats"].get("users", ""),
        stats_tools=info["stats"].get("tools", ""),
        stats_solutions=info["stats"].get("solutions", ""),
    )

    logger.info("Generating platform QA pairs via LLM...")
    try:
        chat = create_chat_provider()
    except Exception as e:
        logger.error("Failed to create chat provider: %s", e)
        return []

    response = chat.chat(
        messages=[ChatMessage(role="user", content=prompt)],
        config=ChatConfig(temperature=0.7, max_tokens=2048),
    )

    # Parse response into QA pairs
    qa_pairs = []
    lines = response.strip().split("\n")
    current_q = ""
    current_a = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Q:") or line.startswith("Q："):
            if current_q and current_a:
                qa_pairs.append(QAPair(
                    question=current_q,
                    answer=current_a.strip(),
                    source_url="/api/generated",
                    confidence=0.7,
                ))
            current_q = line[2:].strip()
            current_a = ""
        elif line.startswith("A:") or line.startswith("A："):
            current_a = line[2:].strip()
        elif current_a and line:
            current_a += " " + line
        elif current_q and line.startswith(("- ", "1.", "2.", "3.")):
            if not current_a:
                current_a = line
            else:
                current_a += "\n" + line

    if current_q and current_a:
        qa_pairs.append(QAPair(
            question=current_q,
            answer=current_a.strip(),
            source_url="/api/generated",
            confidence=0.7,
        ))

    logger.info("Generated %d platform QA pairs", len(qa_pairs))
    return qa_pairs


# ---------------------------------------------------------------------------
# Hardcoded platform navigation QA (authoritative, not LLM-generated)
# ---------------------------------------------------------------------------

def _generate_nav_qa_from_config() -> list[QAPair]:
    """Generate authoritative navigation QA pairs from site config nav_menus.

    These are NOT LLM-generated — they come directly from the site structure.
    """
    config = _load_site_config()
    if not config:
        return []

    nav_raw = config.get("nav_menus", "")
    try:
        menus = json.loads(nav_raw) if isinstance(nav_raw, str) else nav_raw
    except json.JSONDecodeError:
        return []

    qa_pairs = []

    # Site-wide overview
    all_sections = []
    for m in menus:
        if not m.get("visible", True):
            continue
        label = m.get("label", "")
        href = m.get("href", "")
        if label and href:
            all_sections.append("- %s : %s" % (label, "https://cqaip.cn" + href if href.startswith("/") else href))
        for c in m.get("children", []):
            if c.get("visible", True):
                all_sections.append("    - %s : %s" % (c.get("label", ""), c.get("href", "")))

    if all_sections:
        texts = [
            "重庆市人工智能公共服务平台提供以下功能模块：\n%s" % "\n".join(all_sections),
            "平台主要包括：\n%s\n\n您可以通过顶部导航栏访问各个功能。" % "\n".join(all_sections[:15]),
        ]
        for t in texts:
            qa_pairs.append(QAPair(
                question="平台有哪些功能？", answer=t, source_url="/api/site-config", confidence=1.0,
            ))
            qa_pairs.append(QAPair(
                question="这个平台能做什么？", answer=t, source_url="/api/site-config", confidence=1.0,
            ))

    # Per-module QA
    for m in menus:
        if not m.get("visible", True):
            continue
        label = m.get("label", "")
        href = m.get("href", "")
        children = m.get("children", [])

        # Questions about this module
        if label and href:
            url = "https://cqaip.cn" + href if href.startswith("/") else href

            # Navigation Q
            qa_pairs.append(QAPair(
                question="%s在哪里？" % label,
                answer="在网站顶部导航栏找到【%s】，点击即可进入。\n直达链接：%s" % (label, url),
                source_url="/api/site-config", confidence=1.0,
            ))
            qa_pairs.append(QAPair(
                question="怎么进入%s？" % label,
                answer="方法：在顶部菜单栏点击【%s】。\n页面地址：%s" % (label, url),
                source_url="/api/site-config", confidence=1.0,
            ))
            qa_pairs.append(QAPair(
                question="如何找到%s？" % label,
                answer="顶部导航栏 → 点击【%s】即可访问。\n链接：%s" % (label, url),
                source_url="/api/site-config", confidence=1.0,
            ))

        # Questions about children
        for c in children:
            if not c.get("visible", True):
                continue
            clabel = c.get("label", "")
            chref = c.get("href", "")
            if not clabel:
                continue
            chref_full = "https://cqaip.cn" + chref if chref.startswith("/") else chref

            qa_pairs.append(QAPair(
                question="%s在哪里？" % clabel,
                answer="在顶部导航栏找到【%s】→ 点击【%s】。\n直达链接：%s" % (label, clabel, chref_full),
                source_url="/api/site-config", confidence=1.0,
            ))
            qa_pairs.append(QAPair(
                question="怎么找到%s？" % clabel,
                answer="进入路径：顶部菜单【%s】> 【%s】。\n页面：%s" % (label, clabel, chref_full),
                source_url="/api/site-config", confidence=1.0,
            ))

    logger.info("Generated %d authoritative navigation QAs", len(qa_pairs))
    return qa_pairs


def generate_all_platform_knowledge() -> tuple[list[QAPair], list[QAPair]]:
    """Generate both navigation QAs (from config) and platform usage QAs (LLM).

    Returns:
        (navigation_qa_pairs, platform_usage_qa_pairs)
    """
    print("=" * 50)
    print("Platform Knowledge Generator")
    print("=" * 50)

    # Part 1: Navigation QA from site config (authoritative)
    print("\n[1/2] Generating navigation QA from site structure...")
    nav_qa = _generate_nav_qa_from_config()
    print("  -> %d navigation QA pairs" % len(nav_qa))

    # Part 2: Platform usage QA via LLM
    print("\n[2/2] Generating platform usage QA via LLM...")
    platform_qa = generate_platform_qa_pairs()
    print("  -> %d platform usage QA pairs" % len(platform_qa))

    total = len(nav_qa) + len(platform_qa)
    print("\nTotal new platform knowledge: %d QA pairs" % total)
    return nav_qa, platform_qa
