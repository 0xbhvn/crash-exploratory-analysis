#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main module for Crash Game 10× Streak Analysis.

This module contains the main entry point for the application.

Usage:
    python main.py --input games.csv [--multiplier_threshold 10.0] [--window 50] [--test_frac 0.2] [--output_dir ./output] [--percentiles 0.25,0.50,0.75]
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from pathlib import Path

# Import rich logging
from logger_config import setup_logging, console, print_info, print_success, print_warning, print_error, print_panel

# Import local modules
from analyzer import CrashStreakAnalyzer

# Load environment variables from .env file
load_dotenv()

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)

# Setup rich logging
logger = setup_logging()


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Crash Game Streak Analysis')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for considering a multiplier as a hit (default: 10.0)')
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
    parser.add_argument('--percentiles', default='0.25,0.50,0.75',
                        help='Comma-separated list of percentile boundaries for clustering (default: 0.25,0.50,0.75)')

    # Add load_model argument
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model without retraining')
    parser.add_argument('--model_path', default='./output/xgboost_model.pkl',
                        help='Path to the saved model file (used with --load_model)')

    # Add proper time-series mode argument (on by default)
    parser.add_argument('--time_series_mode', action='store_true', default=True,
                        help='Use proper time-series train/test splitting to prevent data leakage (default: True)')
    parser.add_argument('--legacy_mode', action='store_true',
                        help='Use the legacy feature preparation method (may cause data leakage)')

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

    # Parse percentiles from command line
    percentiles = [float(p) for p in args.percentiles.split(',')]

    # Display welcome message
    print_panel(
        f"Crash Game {args.multiplier_threshold}× Streak Analysis",
        title="Welcome",
        style="green"
    )

    # Handle CSV update if requested
    if args.update_csv:
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
            else:
                print_error(f"{fetch_type.capitalize()} data fetch failed")
                if not os.path.exists(args.input):
                    print_error(f"Input file {args.input} not found. Exiting.")
                    sys.exit(1)
                print_warning("Continuing with existing data...")

        except ImportError:
            print_error(
                "Could not import fetch_data module. Make sure fetch_data.py is in the same directory.")
            if not os.path.exists(args.input):
                print_error(f"Input file {args.input} not found. Exiting.")
                sys.exit(1)
            print_warning("Continuing with existing data...")

    logger.info(f"Starting Crash {args.multiplier_threshold}× Streak Analysis with input={args.input}, "
                f"window={args.window}, test_frac={args.test_frac}, percentiles={percentiles}")

    # Initialize analyzer
    analyzer = CrashStreakAnalyzer(
        multiplier_threshold=args.multiplier_threshold,
        window=args.window,
        test_frac=args.test_frac,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        percentiles=percentiles,
        time_series_mode=not args.legacy_mode
    )

    # Load data
    analyzer.load_data(args.input)

    # Check if this is an update run
    if args.update:
        from daily_updates import load_new_data
        print_info(f"Running daily update with new data from {args.update}")
        new_data = load_new_data(args.update, args.multiplier_threshold)
        retrained = analyzer.daily_update(new_data, args.drift_threshold)
        if retrained:
            print_success("Model was retrained due to detected drift")
            analyzer.save_snapshot()
        else:
            print_info("No significant drift detected, model unchanged")
        sys.exit(0)

    # Regular analysis pipeline
    print_info("Running standard streak-based analysis pipeline")

    # Analyze streaks
    streak_lengths = analyzer.analyze_streaks()

    # Plot streaks if requested
    if args.save_plots:
        analyzer.plot_streaks(streak_lengths)

    # Prepare streak-based features
    analyzer.prepare_features()

    # Train model or load existing model
    if args.load_model:
        print_info(
            f"Loading existing model from {args.model_path} without retraining")
        try:
            import joblib
            analyzer.bst_final = joblib.load(args.model_path)
            # Extract feature_cols and other necessary components if needed
            if isinstance(analyzer.bst_final, dict) and "feature_cols" in analyzer.bst_final:
                analyzer.feature_cols = analyzer.bst_final.get("feature_cols")
                print_info(
                    f"Loaded feature columns from model bundle: {len(analyzer.feature_cols)} features")
            if isinstance(analyzer.bst_final, dict) and "scaler" in analyzer.bst_final:
                analyzer.scaler = analyzer.bst_final.get("scaler")
                print_info("Loaded scaler from model bundle")
            if isinstance(analyzer.bst_final, dict) and "percentile_values" in analyzer.bst_final:
                analyzer.percentile_values = analyzer.bst_final.get(
                    "percentile_values")
            print_success("Successfully loaded existing model")
        except Exception as e:
            print_error(f"Error loading model: {str(e)}")
            print_error("Falling back to training a new model")
            analyzer.train_model()
    else:
        # Train model
        analyzer.train_model()

    # Example prediction with most recent streak data
    # The predict_next_cluster method now handles extracting recent streaks if None is provided
    prediction = analyzer.predict_next_cluster()

    # Display prediction
    print_panel(
        f"Example prediction for next streak length cluster: {prediction}",
        title="Streak-Based Prediction Result",
        style="blue"
    )

    print_success("Streak-based analysis complete!")


if __name__ == "__main__":
    main()
