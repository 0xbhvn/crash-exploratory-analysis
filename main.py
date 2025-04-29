#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module for Crash Game 10× Streak Analysis.

This module contains the main entry point for the application.

Usage:
    python main.py --update_csv_only
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from pathlib import Path

# Import rich logging
from utils.logger_config import setup_logging, print_info, print_success, print_warning, print_error, print_panel

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Setup rich logging
logger = setup_logging()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Crash Game Streak Analysis')

    # Flag for database fetch
    parser.add_argument('--update_csv_only', action='store_true',
                        help='Only update the CSV data from the database and exit without running analysis')
    parser.add_argument('--update_csv', action='store_true',
                        help='Update the CSV data from the database before analysis')
    parser.add_argument('--full_fetch', action='store_true',
                        help='Fetch all data instead of only new data (with --update_csv)')
    parser.add_argument('--fetch_limit', type=int,
                        help='Limit the number of rows to fetch from database')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for considering a multiplier as a hit (default: 10.0)')

    return parser.parse_args()


def main():
    """Main function to handle CSV updates only."""
    # Parse command line arguments
    args = parse_arguments()

    # Display welcome message
    print_panel(
        f"Crash Game {args.multiplier_threshold}× Streak Analysis",
        title="Welcome",
        style="green"
    )

    # Handle CSV update if requested
    if args.update_csv or args.update_csv_only:
        print_info("Updating CSV data from database...")
        try:
            from fetch_data import fetch_crash_data, fetch_incremental_data

            if args.full_fetch:
                result = fetch_crash_data(
                    args.input, args.fetch_limit, args.multiplier_threshold)
                fetch_type = "full"
            else:
                result = fetch_incremental_data(
                    args.input, multiplier_threshold=args.multiplier_threshold)
                fetch_type = "incremental"

            if result:
                print_success(
                    f"{fetch_type.capitalize()} data fetch completed successfully")
                if args.update_csv_only:
                    print_success("CSV update complete. Exiting as requested.")
                    sys.exit(0)
            else:
                print_error(f"{fetch_type.capitalize()} data fetch failed")
                if not os.path.exists(args.input):
                    print_error(f"Input file {args.input} not found. Exiting.")
                    sys.exit(1)
                if args.update_csv_only:
                    print_error("CSV update failed. Exiting.")
                    sys.exit(1)
                print_warning("Continuing with existing data...")

        except ImportError:
            print_error(
                "Could not import fetch_data module. Make sure fetch_data.py is in the same directory.")
            if not os.path.exists(args.input):
                print_error(f"Input file {args.input} not found. Exiting.")
                sys.exit(1)
            if args.update_csv_only:
                print_error("CSV update failed. Exiting.")
                sys.exit(1)


if __name__ == "__main__":
    main()
