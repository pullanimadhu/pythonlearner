#!/usr/bin/env python3
"""CLI script to build the RAG indices from CSV data."""

import sys

from app.data.indexer import build_index
from app.utils.logger import logger


def main() -> None:
    logger.info("Starting index build...")
    try:
        build_index()
        logger.info("Index build completed successfully.")
    except Exception as e:
        logger.error("Index build failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
