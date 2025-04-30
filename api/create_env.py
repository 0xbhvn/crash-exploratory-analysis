#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create a .env file for the API.

This script creates a .env file with default values.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_env_file(output_file=".env", overwrite=False):
    """
    Create a .env file with default values.

    Args:
        output_file: Path to the output .env file
        overwrite: Whether to overwrite an existing file
    """
    output_path = Path(output_file)

    # Check if file already exists
    if output_path.exists() and not overwrite:
        logger.warning(
            f"{output_file} already exists. Use --overwrite to replace it.")
        return False

    # Define default environment variables
    env_vars = {
        "# Database connection": "",
        "DATABASE_URL": "postgresql://postgres:postgres@localhost:5432/crash_game",
        "POSTGRES_USER": "postgres",
        "POSTGRES_PASSWORD": "postgres",
        "POSTGRES_DB": "crash_game",
        "": "",
        "# API configuration": "",
        "API_URL": "http://localhost:8000",
        "MULTIPLIER_THRESHOLD": "10.0",
        "PREDICTION_CONFIDENCE_THRESHOLD": "0.75",
        "COLLECTOR_SLEEP_SECONDS": "30",
        "": "",
        "# WebSocket configuration": "",
        "EXTERNAL_WS_URL": "wss://crashed-proxy-production.up.railway.app/ws",
        "": "",
        "# Notification endpoints (optional)": "",
        "SLACK_WEBHOOK": "",
        "TELEGRAM_BOT_TOKEN": "",
        "TELEGRAM_CHAT_ID": ""
    }

    # Write to file
    with open(output_path, 'w') as f:
        for key, value in env_vars.items():
            if key.startswith("#") or not key:
                f.write(f"{key}\n")
            else:
                f.write(f"{key}={value}\n")

    logger.info(f"Created .env file at {output_path}")
    return True


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Create a .env file with default values")
    parser.add_argument("--output", default=".env", help="Output file path")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing file")
    args = parser.parse_args()

    # Create .env file
    success = create_env_file(args.output, args.overwrite)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
