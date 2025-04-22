#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for Crash Game 10× Streak Analysis.

This script provides the command-line interface and coordinates the analysis.

Usage:
    python main.py --input games.csv [--window 50] [--test_frac 0.2] [--output_dir ./output]
"""

from analyzer import CrashStreakAnalyzer
import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/crash_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Import local modules


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Crash Game 10× Streak Analysis')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--window', type=int, default=50,
                        help='Rolling window size for feature engineering')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--output_dir', default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_plots', action='store_true',
                        help='Generate and save plots')
    parser.add_argument('--update',
                        help='Path to new data file for model update')
    parser.add_argument('--drift_threshold', type=float, default=0.005,
                        help='Threshold for detecting distribution drift')

    # Flags for database fetch
    parser.add_argument('--update_csv', action='store_true',
                        help='Update the CSV data from the database before analysis')
    parser.add_argument('--full_fetch', action='store_true',
                        help='Fetch all data instead of only new data (with --update_csv)')
    parser.add_argument('--fetch_limit', type=int,
                        help='Limit the number of rows to fetch from database')

    return parser.parse_args()


def main():
    """
    Main function to run the crash streak analysis pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Handle CSV update if requested
    if args.update_csv:
        logger.info("Updating CSV data from database...")
        try:
            from fetch_data import fetch_crash_data, fetch_incremental_data

            if args.full_fetch:
                result = fetch_crash_data(args.input, args.fetch_limit)
                fetch_type = "full"
            else:
                result = fetch_incremental_data(args.input)
                fetch_type = "incremental"

            if result:
                logger.info(
                    f"✅ {fetch_type.capitalize()} data fetch completed successfully")
            else:
                logger.error(f"❌ {fetch_type.capitalize()} data fetch failed")
                if not os.path.exists(args.input):
                    logger.error(
                        f"Input file {args.input} not found. Exiting.")
                    sys.exit(1)
                logger.warning("Continuing with existing data...")

        except ImportError:
            logger.error(
                "Could not import fetch_data module. Make sure fetch_data.py is in the same directory.")
            if not os.path.exists(args.input):
                logger.error(f"Input file {args.input} not found. Exiting.")
                sys.exit(1)
            logger.warning("Continuing with existing data...")

    logger.info(f"Starting Crash 10× Streak Analysis with input={args.input}, "
                f"window={args.window}, test_frac={args.test_frac}")

    # Initialize analyzer
    analyzer = CrashStreakAnalyzer(
        window=args.window,
        test_frac=args.test_frac,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )

    # Load data
    analyzer.load_data(args.input)

    # Check if this is an update run
    if args.update:
        from daily_updates import load_new_data
        new_data = load_new_data(args.update)
        retrained = analyzer.daily_update(new_data, args.drift_threshold)
        if retrained:
            logger.info("Model was retrained due to detected drift")
            analyzer.save_snapshot()
        sys.exit(0)

    # Regular analysis pipeline
    # Analyze streaks
    streak_lengths = analyzer.analyze_streaks()

    # Plot streaks if requested
    if args.save_plots:
        analyzer.plot_streaks(streak_lengths)

    # Prepare features
    analyzer.prepare_features()

    # Train model
    analyzer.train_model()

    # Example prediction with last window
    demo_input = analyzer.df["Bust"].iloc[-analyzer.WINDOW:].tolist()
    prediction = analyzer.predict_next_cluster(demo_input)

    logger.info(
        f"Example prediction for next streak length cluster: {prediction}")
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
