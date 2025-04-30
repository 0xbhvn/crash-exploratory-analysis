#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run the API for testing.

This script provides a convenient way to run the API for development and testing.
"""

import os
import sys
import argparse
import logging
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)
# Add parent directory to find the temporal module
sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the API."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run the API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload")
    args = parser.parse_args()

    # Run the API
    logger.info(f"Starting API server on {args.host}:{args.port}...")
    logger.info(f"Python path includes parent directory: {parent_dir}")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
