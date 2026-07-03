"""Data source definitions for cqaip.cn API endpoints.

Each source describes an API endpoint and how to extract data from it.
"""

from dataclasses import dataclass
from typing import Callable, Literal

from shared.enums import DocumentSource


@dataclass
class Pagination:
    """How to paginate through a list endpoint."""

    type: Literal["page", "offset", "none"]
    param_page: str = "page"
    param_size: str = "pageSize"
    page_size: int = 20
    # JSON path to total count (needed for paginated endpoints)
    total_path: list[str] | None = None  # e.g., ["data", "total"]


@dataclass
class DataSource:
    """Configuration for a single API data source."""

    name: str
    endpoint: str
    doc_source: DocumentSource
    # Whether this endpoint requires the Casdoor session cookie
    requires_auth: bool = False
    # Path to the data array inside the JSON response
    # e.g., ["data"] for {"code":200, "data": [...]}
    # e.g., ["data", "items"] for {"code":200, "data": {"items": [...]}}
    data_path: list[str] | None = None
    # Pagination config (None = no pagination, fetch once)
    pagination: Pagination | None = None
    # Category label for documents extracted from this source
    category: str = ""


# ---- All discovered data sources ----

SOURCES: list[DataSource] = [
    # ---- Public endpoints ----
    DataSource(
        name="policies",
        endpoint="/api/policies",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="政策",
    ),
    DataSource(
        name="industry_news",
        endpoint="/api/industry-news",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data", "items"],
        pagination=Pagination(
            type="page",
            page_size=15,
            total_path=["data", "total"],
        ),
        category="行业资讯",
    ),
    DataSource(
        name="competitions",
        endpoint="/api/competitions",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="赛事活动",
    ),
    DataSource(
        name="communities",
        endpoint="/api/communities",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="社区",
    ),
    DataSource(
        name="news",
        endpoint="/api/news",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="新闻",
    ),
    # ---- Authenticated endpoints (need casdoor_session_id) ----
    DataSource(
        name="datasets",
        endpoint="/api/datasets",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data", "items"],
        pagination=Pagination(
            type="page",
            page_size=20,
            total_path=["data", "total"],
        ),
        requires_auth=True,
        category="数据集",
    ),
    DataSource(
        name="skills",
        endpoint="/api/skills",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data", "items"],
        pagination=Pagination(
            type="page",
            page_size=20,
            total_path=["data", "total"],
        ),
        requires_auth=True,
        category="技能",
    ),
    DataSource(
        name="marketplace",
        endpoint="/api/marketplace",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data", "items"],
        pagination=Pagination(
            type="page",
            page_size=12,
            total_path=["data", "total"],
        ),
        requires_auth=True,
        category="供需市场",
    ),
    # ---- Metadata endpoints (single item) ----
    DataSource(
        name="models",
        endpoint="/api/models",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data", "items"],
        pagination=Pagination(
            type="page",
            page_size=20,
            total_path=["data", "total"],
        ),
        category="模型",
    ),
    DataSource(
        name="apps",
        endpoint="/api/apps",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="应用",
    ),
    DataSource(
        name="site_config",
        endpoint="/api/site-config",
        doc_source=DocumentSource.CQAIP_HTML,
        data_path=["data"],
        category="站点配置",
    ),
]

# Lookup by name
SOURCES_BY_NAME = {s.name: s for s in SOURCES}
