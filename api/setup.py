#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for the API.

This script sets up the API project:
1. Creates an .env file if one doesn't exist
2. Runs database migrations
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_env_file():
    """Create a .env file if one doesn't exist."""
    try:
        env_path = Path(".env")
        if not env_path.exists():
            logger.info("Creating .env file with default values...")
            # Use the create_env.py script
            result = subprocess.run(
                [sys.executable, "create_env.py"],
                check=True
            )
            if result.returncode != 0:
                logger.error("Failed to create .env file")
                return False
            logger.info(".env file created successfully")
        else:
            logger.info(".env file already exists")
        return True
    except Exception as e:
        logger.error(f"Error creating .env file: {str(e)}")
        return False


def run_migrations():
    """Run database migrations."""
    try:
        logger.info("Running database migrations...")
        # Use the init_db.py script
        result = subprocess.run(
            [sys.executable, "init_db.py"],
            check=True
        )
        if result.returncode != 0:
            logger.error("Failed to run migrations")
            return False
        logger.info("Migrations completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False


def main():
    """Main function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Set up the API project")
    parser.add_argument("--skip-env", action="store_true",
                        help="Skip creating .env file")
    parser.add_argument("--skip-migrations",
                        action="store_true", help="Skip running migrations")
    args = parser.parse_args()

    # Create .env file
    if not args.skip_env:
        if not create_env_file():
            logger.error("Failed to create .env file")
            sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Run migrations
    if not args.skip_migrations:
        if not run_migrations():
            logger.error("Failed to run migrations")
            sys.exit(1)

    logger.info("Setup completed successfully")


if __name__ == "__main__":
    main()
