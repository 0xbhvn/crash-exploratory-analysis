#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data fetching module for Crash Game 10× Streak Analysis.

This script connects to the database and downloads game data to a CSV file.
Only the game_id and crash_point columns are extracted.
"""

import os
import logging
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging for standalone execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/data_fetch.log')
        ]
    )


def fetch_crash_data(output_file: str = 'games.csv', limit: Optional[int] = None) -> bool:
    """
    Fetch crash game data from the database and save to CSV.

    Args:
        output_file: Path to save the CSV file
        limit: Optional limit on number of rows to fetch

    Returns:
        Boolean indicating success
    """
    # Get database connection string from environment variable
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error(
            "DATABASE_URL environment variable not set in environment or .env file")
        return False

    try:
        # Create database engine
        logger.info("Connecting to database...")
        engine = create_engine(database_url)

        # Prepare query
        query = "SELECT game_id as \"Game ID\", crash_point as \"Bust\" FROM crash_games ORDER BY game_id DESC"
        if limit:
            query += f" LIMIT {limit}"

        # Execute query
        logger.info("Executing database query...")
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)

        # Save to CSV
        logger.info(f"Saving {len(df)} rows to {output_file}")
        df.to_csv(output_file, index=False)

        logger.info("Data fetch complete")
        return True

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return False


def fetch_incremental_data(output_file: str = 'games.csv',
                           last_game_id: Optional[int] = None) -> bool:
    """
    Fetch only new crash game data since the last known game_id.

    Args:
        output_file: Path to save or update the CSV file
        last_game_id: The highest game_id already in the dataset

    Returns:
        Boolean indicating success
    """
    # Determine last_game_id from existing file if not provided
    if last_game_id is None and os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            if not existing_df.empty and "Game ID" in existing_df.columns:
                last_game_id = existing_df["Game ID"].max()
                logger.info(f"Last game ID in existing data: {last_game_id}")
        except Exception as e:
            logger.warning(f"Could not read existing file: {str(e)}")

    # Get database connection string from environment variable
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error(
            "DATABASE_URL environment variable not set in environment or .env file")
        return False

    try:
        # Create database engine
        logger.info("Connecting to database...")
        engine = create_engine(database_url)

        # Prepare query for new data
        query = "SELECT game_id as \"Game ID\", crash_point as \"Bust\" FROM crash_games"
        if last_game_id:
            query += f" WHERE game_id > {last_game_id}"
        query += " ORDER BY game_id DESC"

        # Execute query
        logger.info("Executing database query for new data...")
        with engine.connect() as conn:
            new_df = pd.read_sql(text(query), conn)

        if new_df.empty:
            logger.info("No new data found")
            return True

        logger.info(f"Fetched {len(new_df)} new rows")

        # Append or create file
        if os.path.exists(output_file) and last_game_id is not None:
            logger.info(f"Appending new data to {output_file}")
            # Append without headers
            new_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            logger.info(f"Creating new file {output_file}")
            new_df.to_csv(output_file, index=False)

        logger.info("Incremental data fetch complete")
        return True

    except Exception as e:
        logger.error(f"Error fetching incremental data: {str(e)}")
        return False


if __name__ == '__main__':
    # If run directly, set up logging and fetch data
    setup_logging()

    import argparse
    parser = argparse.ArgumentParser(
        description='Fetch crash game data from database')
    parser.add_argument('--output', default='games.csv',
                        help='Output CSV file')
    parser.add_argument('--limit', type=int,
                        help='Limit number of rows to fetch')
    parser.add_argument('--incremental', action='store_true',
                        help='Fetch only new data since last known game_id')
    args = parser.parse_args()

    if args.incremental:
        result = fetch_incremental_data(args.output)
    else:
        result = fetch_crash_data(args.output, args.limit)

    if result:
        logger.info("✅ Data fetch successful")
    else:
        logger.error("❌ Data fetch failed")
        exit(1)
