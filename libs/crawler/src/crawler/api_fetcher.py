"""Generic REST API fetcher for cqaip.cn data sources.

Handles pagination, authentication, retries, and saves raw JSON data.
Completely replaces Scrapy for this project - all data comes from APIs.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from config.settings import settings

from .sources import SOURCES, DataSource

logger = logging.getLogger(__name__)


def _deep_get(data: dict, path: list[str] | None) -> Any:
    """Access a nested dict value by path list.

    >>> data = {"code": 200, "data": {"items": [1, 2, 3]}}
    >>> _deep_get(data, ["data", "items"])
    [1, 2, 3]
    """
    if not path:
        return data
    result = data
    for key in path:
        if isinstance(result, dict):
            result = result.get(key)
        else:
            return None
    return result


def _get_total(data: dict, source: DataSource) -> int | None:
    """Extract total count from a paginated response."""
    if source.pagination and source.pagination.total_path:
        return _deep_get(data, source.pagination.total_path)
    return None


class APIFetcher:
    """Fetch data from all configured REST API sources.

    Usage:
        fetcher = APIFetcher()
        results = fetcher.fetch_all()
        # or fetcher.fetch_one("policies")
    """

    def __init__(self):
        self.base_url = settings.api_base_url.rstrip("/")
        self.timeout = settings.request_timeout
        self.max_retries = settings.request_retries
        self.auth_cookie = (
            settings.casdoor_session_id
            if settings.casdoor_session_id
            else None
        )
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            headers = {
                "User-Agent": settings.crawl_user_agent,
                "Accept": "application/json",
            }
            cookies = {}
            if self.auth_cookie:
                cookies["casdoor_session_id"] = self.auth_cookie
            self._client = httpx.Client(
                headers=headers,
                cookies=cookies,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    def _request(self, url: str) -> dict:
        """Make an HTTP GET request with retries."""
        client = self._get_client()
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if attempt > 1:
                    delay = 2 ** attempt  # exponential backoff
                    logger.debug("Retry %d/%d after %ds: %s", attempt, self.max_retries, delay, url)
                    time.sleep(delay)
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning("HTTP %d for %s", e.response.status_code, url)
                if e.response.status_code in (401, 403):
                    logger.error("Auth failed for %s - check casdoor_session_id", url)
                    raise
                if e.response.status_code == 404:
                    logger.warning("Endpoint not found: %s", url)
                    raise
            except httpx.RequestError as e:
                last_error = e
                logger.warning("Request failed (attempt %d): %s", attempt, e)
            except json.JSONDecodeError as e:
                last_error = e
                logger.error("Invalid JSON from %s: %s", url, e)
                raise
        raise RuntimeError(
            "All %d retries failed for %s: %s" % (self.max_retries, url, last_error)
        )

    def _fetch_page(self, source: DataSource, page: int = 1) -> dict:
        """Fetch a single page of data from a source."""
        sep = "&" if "?" in source.endpoint else "?"
        if source.pagination and source.pagination.type in ("page", "offset"):
            url = (
                "%s%s%s%d&%s=%d"
                % (
                    self.base_url,
                    source.endpoint,
                    sep + source.pagination.param_page + "=",
                    page,
                    source.pagination.param_size,
                    source.pagination.page_size,
                )
            )
        else:
            url = "%s%s" % (self.base_url, source.endpoint)
        return self._request(url)

    def fetch_source(self, source: DataSource) -> list[dict]:
        """Fetch all items from a single data source (handles pagination)."""
        logger.info("Fetching: %s (%s)", source.name, source.endpoint)

        all_items: list[dict] = []

        if source.pagination is None or source.pagination.type == "none":
            # Non-paginated - single request
            data = self._fetch_page(source)
            items = _deep_get(data, source.data_path) or []
            if isinstance(items, dict):
                all_items = [items]
            elif isinstance(items, list):
                all_items = items
        else:
            # Paginated - fetch all pages
            page = 1
            while True:
                data = self._fetch_page(source, page)
                items = _deep_get(data, source.data_path) or []
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    all_items.append(items)

                total = _get_total(data, source)
                page_size = source.pagination.page_size
                if total is not None and page * page_size >= total:
                    break
                # Also stop if fewer items returned than page size
                if isinstance(items, list) and len(items) < page_size:
                    break
                page += 1

        logger.info("  -> %s: %d items fetched", source.name, len(all_items))
        return all_items

    def fetch_all(self, source_names: list[str] | None = None) -> dict[str, list[dict]]:
        """Fetch all (or specified) data sources.

        Returns:
            Dict mapping source name -> list of raw item dicts.
        """
        targets = (
            [s for s in SOURCES if s.name in source_names]
            if source_names
            else SOURCES
        )
        results = {}
        for source in targets:
            try:
                results[source.name] = self.fetch_source(source)
            except Exception as e:
                logger.error("Failed to fetch %s: %s", source.name, e)
                results[source.name] = []
        return results

    def save_raw(self, source_name: str, items: list[dict]) -> Path:
        """Save fetched items to a raw JSON file.

        Returns the file path.
        """
        raw_dir = Path(settings.raw_data_dir) / source_name
        raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = raw_dir / ("%s.json" % timestamp)

        payload = {
            "source": source_name,
            "fetched_at": datetime.now().isoformat(),
            "count": len(items),
            "items": items,
        }
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("  -> Saved %d items to %s", len(items), file_path)
        return file_path

    def fetch_and_save_all(
        self, source_names: list[str] | None = None
    ) -> dict[str, Path]:
        """Fetch all sources and save them to disk.

        Returns:
            Dict mapping source name -> saved file path.
        """
        results = self.fetch_all(source_names)
        saved = {}
        for name, items in results.items():
            if items:
                saved[name] = self.save_raw(name, items)
            else:
                logger.warning("  -> %s: 0 items, nothing saved", name)
        return saved

    def close(self):
        if self._client:
            self._client.close()
            self._client = None
