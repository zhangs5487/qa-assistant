"""Run the full data pipeline: fetch -> clean -> chunk.

Usage:
    python scripts/run_pipeline.py                   # Fetch all + clean all
    python scripts/run_pipeline.py --fetch-only       # Only fetch from APIs
    python scripts/run_pipeline.py --clean-only       # Only clean existing raw data
    python scripts/run_pipeline.py --sources policies,datasets  # Specific sources
    python scripts/run_pipeline.py --no-auth          # Skip authenticated endpoints
"""

import argparse
import logging
import sys
import os

# Ensure packages are importable
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_BASE, "libs", "crawler", "src"))
sys.path.insert(0, os.path.join(_BASE, "libs", "cleaner", "src"))
sys.path.insert(0, os.path.join(_BASE, "libs", "shared", "src"))
sys.path.insert(0, os.path.join(_BASE, "config"))


def main():
    parser = argparse.ArgumentParser(description="QA Assistant - Data Pipeline")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data from APIs")
    parser.add_argument("--clean-only", action="store_true", help="Only clean existing raw data")
    parser.add_argument("--sources", type=str, default="", help="Comma-separated source names")
    parser.add_argument("--no-auth", action="store_true", help="Skip authenticated endpoints")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine source names
    source_names = [s.strip() for s in args.sources.split(",")] if args.sources else None

    do_fetch = not args.clean_only
    do_clean = not args.fetch_only

    # Step 1: Fetch
    if do_fetch:
        from crawler.api_fetcher import APIFetcher
        from crawler.sources import SOURCES

        fetcher = APIFetcher()

        # Filter sources if --no-auth or --sources
        targets = SOURCES
        if args.no_auth:
            targets = [s for s in targets if not s.requires_auth]
        if source_names:
            targets = [s for s in targets if s.name in source_names]

        print("=" * 50)
        print("Step 1: Fetching %d data sources..." % len(targets))
        print("=" * 50)

        saved = fetcher.fetch_and_save_all([s.name for s in targets])
        fetcher.close()

        print("\nFetch complete: %d sources" % len(saved))
        for name, path in sorted(saved.items()):
            print("  [OK] %s: %s" % (name, path))

    # Step 2: Clean + Chunk
    if do_clean:
        from cleaner.pipeline import process_all
        from pathlib import Path

        print("\n" + "=" * 50)
        print("Step 2: Cleaning and chunking...")
        print("=" * 50)

        results = process_all(source_names=source_names)

        total_docs = sum(v[0] for v in results.values())
        total_chunks = sum(v[1] for v in results.values())

        print("\nClean complete:")
        for name, (docs, chunks) in sorted(results.items()):
            print("  [OK] %s: %d docs, %d chunks" % (name, docs, chunks))
        print("\nTOTAL: %d documents, %d chunks" % (total_docs, total_chunks))


if __name__ == "__main__":
    main()
