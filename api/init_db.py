#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database initialization script.

This script initializes the database with tables and initial data.
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_migrations():
    """Run Alembic migrations."""
    try:
        logger.info("Running Alembic migrations...")
        # Check if versions directory exists
        versions_dir = Path("migrations/versions")
        versions_dir.mkdir(exist_ok=True)

        # Check if we need to initialize
        if not list(versions_dir.glob("*.py")):
            logger.info(
                "No migration files found. Creating initial migration...")
            result = subprocess.run(
                ["alembic", "revision", "--autogenerate", "-m", "Initial migration"],
                check=True
            )
            if result.returncode != 0:
                logger.error("Failed to create initial migration")
                return False

        # Run migrations
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            check=True
        )
        if result.returncode != 0:
            logger.error("Failed to apply migrations")
            return False

        logger.info("Migrations applied successfully")
        return True
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        return False


def main():
    """Main function for database initialization."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Initialize database and run migrations")
    parser.add_argument("--skip-migrations", action="store_true",
                        help="Skip running Alembic migrations")
    args = parser.parse_args()

    # Run migrations
    if not args.skip_migrations:
        if not run_migrations():
            logger.error("Database initialization failed")
            sys.exit(1)

    logger.info("Database initialization completed successfully")


if __name__ == "__main__":
    main()
