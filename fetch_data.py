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
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from pathlib import Path

# Import rich logging
from logger_config import (
    setup_logging, console, create_table, display_table,
    add_table_row, create_stats_table, print_info, print_success,
    print_warning, print_error, print_panel
)

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


def setup_logging():
    """Set up logging for standalone execution."""
    return setup_logging(log_file='logs/data_fetch.log')


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
        print_error(
            "DATABASE_URL environment variable not set in environment or .env file")
        return False

    try:
        # Create database engine
        print_info("Connecting to database...")
        engine = create_engine(database_url)

        # Prepare query
        query = "SELECT game_id as \"Game ID\", crash_point as \"Bust\" FROM crash_games ORDER BY game_id ASC"
        if limit:
            query += f" LIMIT {limit}"
            print_info(f"Query will be limited to {limit} rows")

        # Execute query
        print_info("Executing database query...")
        with engine.connect() as conn:
            # Get start time for query execution timing
            start_time = datetime.now()

            df = pd.read_sql(text(query), conn)

            # Calculate query execution time
            execution_time = (datetime.now() - start_time).total_seconds()

        # Display statistics as a table
        stats = {
            "Total Rows": len(df),
            "First Game ID": df["Game ID"].min() if not df.empty else "N/A",
            "Last Game ID": df["Game ID"].max() if not df.empty else "N/A",
            "Min Multiplier": df["Bust"].min() if not df.empty else "N/A",
            "Max Multiplier": df["Bust"].max() if not df.empty else "N/A",
            "Avg Multiplier": df["Bust"].mean() if not df.empty else "N/A",
            "10× or Higher": (df["Bust"] >= 10).sum() if not df.empty else "N/A",
            "10× Rate": f"{(df['Bust'] >= 10).mean() * 100:.2f}%" if not df.empty else "N/A",
            "Query Time": f"{execution_time:.2f} seconds"
        }

        create_stats_table("Database Fetch Results", stats)

        # Show sample of the data
        if not df.empty:
            sample_table = create_table("Sample Data", ["Game ID", "Bust"])
            for _, row in df.head(5).iterrows():
                add_table_row(
                    sample_table, [row["Game ID"], f"{row['Bust']:.2f}"])
            display_table(sample_table)

        # Save to CSV
        print_info(f"Saving {len(df):,} rows to {output_file}")
        df.to_csv(output_file, index=False)

        print_success("Data fetch complete")
        return True

    except Exception as e:
        print_error(f"Error fetching data: {str(e)}")
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
                print_info(f"Last game ID in existing data: {last_game_id}")
        except Exception as e:
            print_warning(f"Could not read existing file: {str(e)}")

    # Get database connection string from environment variable
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print_error(
            "DATABASE_URL environment variable not set in environment or .env file")
        return False

    try:
        # Create database engine
        print_info("Connecting to database...")
        engine = create_engine(database_url)

        # Prepare query for new data
        query = "SELECT game_id as \"Game ID\", crash_point as \"Bust\" FROM crash_games"
        if last_game_id:
            query += f" WHERE CAST(game_id AS BIGINT) > {last_game_id}"
        query += " ORDER BY game_id ASC"

        # Execute query
        print_info("Executing database query for new data...")
        with engine.connect() as conn:
            # Get start time for query execution timing
            start_time = datetime.now()

            new_df = pd.read_sql(text(query), conn)

            # Calculate query execution time
            execution_time = (datetime.now() - start_time).total_seconds()

        if new_df.empty:
            print_info("No new data found")
            return True

        # Display statistics in a table
        stats = {
            "New Rows": len(new_df),
            "First New Game ID": new_df["Game ID"].min() if not new_df.empty else "N/A",
            "Last New Game ID": new_df["Game ID"].max() if not new_df.empty else "N/A",
            "Min Multiplier": new_df["Bust"].min() if not new_df.empty else "N/A",
            "Max Multiplier": new_df["Bust"].max() if not new_df.empty else "N/A",
            "Avg Multiplier": new_df["Bust"].mean() if not new_df.empty else "N/A",
            "10× or Higher": (new_df["Bust"] >= 10).sum() if not new_df.empty else "N/A",
            "10× Rate": f"{(new_df['Bust'] >= 10).mean() * 100:.2f}%" if not new_df.empty else "N/A",
            "Query Time": f"{execution_time:.2f} seconds"
        }

        create_stats_table("Incremental Data Fetch Results", stats)

        # Show sample of the new data
        if not new_df.empty:
            sample_table = create_table("Sample New Data", ["Game ID", "Bust"])
            for _, row in new_df.head(5).iterrows():
                add_table_row(
                    sample_table, [row["Game ID"], f"{row['Bust']:.2f}"])
            display_table(sample_table)

        # Append or create file
        if os.path.exists(output_file) and last_game_id is not None:
            print_info(f"Appending {len(new_df):,} new rows to {output_file}")
            # Append without headers
            new_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            print_info(
                f"Creating new file {output_file} with {len(new_df):,} rows")
            new_df.to_csv(output_file, index=False)

        print_success("Incremental data fetch complete")
        return True

    except Exception as e:
        print_error(f"Error fetching incremental data: {str(e)}")
        return False


if __name__ == '__main__':
    # If run directly, set up logging and fetch data
    setup_logging()

    # Display welcome panel
    print_panel(
        "This tool fetches crash game data from the database",
        title="Crash Game Data Fetcher",
        style="green"
    )

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
        print_success("Data fetch successful")
    else:
        print_error("Data fetch failed")
        exit(1)
