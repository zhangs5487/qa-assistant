"""CLI entry point for the API fetcher.

Usage:
    python -m crawler.cli --all                  # Fetch all sources
    python -m crawler.cli --sources policies,news # Fetch specific sources
    python -m crawler.cli --list                  # List available sources
"""

import argparse
import logging
import sys

from .sources import SOURCES, SOURCES_BY_NAME
from .api_fetcher import APIFetcher


def main():
    parser = argparse.ArgumentParser(description="cqaip.cn API data fetcher")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all available data sources",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="",
        help="Comma-separated source names to fetch",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data sources and exit",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list:
        print("Available data sources:")
        for s in SOURCES:
            auth = " [auth]" if s.requires_auth else ""
            pages = " (paginated)" if s.pagination else ""
            print("  %-20s %-30s%s%s" % (s.name, s.endpoint, auth, pages))
        return

    if not args.all and not args.sources:
        parser.print_help()
        sys.exit(1)

    if args.sources:
        names = [n.strip() for n in args.sources.split(",")]
        unknown = [n for n in names if n not in SOURCES_BY_NAME]
        if unknown:
            print("Unknown sources: " + str(unknown))
            print("Available: " + str(list(SOURCES_BY_NAME.keys())))
            sys.exit(1)
    else:
        names = None

    fetcher = APIFetcher()
    try:
        saved = fetcher.fetch_and_save_all(names)
        print("\nDone! Fetched %d sources:" % len(saved))
        for name, path in sorted(saved.items()):
            print("  [OK] %s: %s" % (name, path))
    except Exception as e:
        print("\nFatal error: %s" % e)
        sys.exit(1)
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
