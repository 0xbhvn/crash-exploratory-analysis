#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict-only script for Crash Game Streak Analysis.

This script loads an existing model and makes predictions on new data without retraining.

Usage:
    python predict_only.py --input games.csv --model_path ./output/xgboost_model.pkl [--update_csv]
"""

from analyzer import CrashStreakAnalyzer
import os
import sys
import argparse
import logging
import joblib
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

# Import rich logging
from logger_config import setup_logging, console, print_info, print_success, print_warning, print_error, print_panel

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
        description='Crash Game Streak Analysis - Prediction Only')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for considering a multiplier as a hit (default: 10.0)')
    parser.add_argument('--window', type=int, default=50,
                        help='Rolling window size for feature engineering')
    parser.add_argument('--output_dir', default='./output',
                        help='Directory to save outputs')
    parser.add_argument('--model_path', default='./output/xgboost_model.pkl',
                        help='Path to the saved model file')
    parser.add_argument('--percentiles', default='0.25,0.50,0.75',
                        help='Comma-separated list of percentile boundaries for clustering (default: 0.25,0.50,0.75)')

    # Flags for database fetch
    parser.add_argument('--update_csv', action='store_true',
                        help='Update the CSV data from the database before analysis')
    parser.add_argument('--fetch_limit', type=int,
                        help='Limit the number of rows to fetch from database')
    parser.add_argument('--streak_count', type=int, default=None,
                        help='Number of recent streaks to use for prediction (default: window size)')

    return parser.parse_args()


def main():
    """
    Main function to run the crash streak prediction pipeline.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Parse percentiles from command line
    percentiles = [float(p) for p in args.percentiles.split(',')]

    # Display welcome message
    print_panel(
        f"Crash Game {args.multiplier_threshold}× Streak Prediction (No Training)",
        title="Welcome",
        style="green"
    )

    # Handle CSV update if requested
    if args.update_csv:
        print_info("Updating CSV data from database...")
        try:
            from fetch_data import fetch_incremental_data
            result = fetch_incremental_data(
                args.input, multiplier_threshold=args.multiplier_threshold)

            if result:
                print_success("Incremental data fetch completed successfully")
            else:
                print_error("Incremental data fetch failed")
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

    # Check if model exists
    if not os.path.exists(args.model_path):
        print_error(f"Model file {args.model_path} not found. Exiting.")
        sys.exit(1)

    # Initialize analyzer with minimal config (we'll load more from the model)
    analyzer = CrashStreakAnalyzer(
        multiplier_threshold=args.multiplier_threshold,
        window=args.window,
        output_dir=args.output_dir,
        percentiles=percentiles
    )

    # Load data
    analyzer.load_data(args.input)

    # Load the existing model
    print_info(f"Loading model from {args.model_path}")
    try:
        model_bundle = joblib.load(args.model_path)
        analyzer.bst_final = model_bundle

        # Extract components from model bundle
        if isinstance(model_bundle, dict):
            # Update analyzer with model parameters
            if "feature_cols" in model_bundle:
                analyzer.feature_cols = model_bundle["feature_cols"]
                print_info(
                    f"Loaded {len(analyzer.feature_cols)} feature columns from model")

            if "scaler" in model_bundle:
                analyzer.scaler = model_bundle["scaler"]
                print_info("Loaded feature scaler from model")

            if "percentile_values" in model_bundle:
                analyzer.percentile_values = model_bundle["percentile_values"]
                print_info(
                    f"Loaded percentile values from model: {analyzer.percentile_values}")

            if "window" in model_bundle:
                # Update window size from model if it differs
                model_window = model_bundle["window"]
                if model_window != args.window:
                    print_warning(
                        f"Overriding window size from command line ({args.window}) with model value ({model_window})")
                    analyzer.WINDOW = model_window

            print_success("Successfully loaded model bundle")
        else:
            print_warning(
                "Model is not a bundle with metadata. Some functionality may be limited.")
    except Exception as e:
        print_error(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Extract streaks from raw data
    from data_processing import extract_streaks_and_multipliers
    streak_df = extract_streaks_and_multipliers(
        analyzer.df, analyzer.MULTIPLIER_THRESHOLD)
    analyzer.streak_df = streak_df
    print_info(f"Extracted {len(streak_df)} streaks from input data")

    # Get the most recent streaks based on window size or streak_count if provided
    streak_count = args.streak_count or analyzer.WINDOW
    recent_streaks = streak_df.tail(streak_count).to_dict('records')

    # Display the most recent streaks
    print_info(
        f"Using {len(recent_streaks)} most recent streaks for prediction")

    # Create a table with recent streak information
    from logger_config import create_table, add_table_row, display_table
    recent_table = create_table("Recent Streaks for Prediction",
                                ["Streak #", "Game ID Range", "Length", "Hit Multiplier"])

    # Show a few recent streaks (last 5 or fewer if we have less)
    for streak in list(reversed(recent_streaks))[:5]:
        add_table_row(recent_table, [
            streak['streak_number'],
            f"{streak['start_game_id']} → {streak['end_game_id']}",
            streak['streak_length'],
            f"{streak['hit_multiplier']:.2f}"
        ])
    display_table(recent_table)

    # Make prediction
    prediction = analyzer.predict_next_cluster(recent_streaks)

    # Sort prediction by probability
    sorted_pred = sorted(prediction.items(),
                         key=lambda x: float(x[1]), reverse=True)

    # Generate cluster descriptions
    cluster_descriptions = {}
    if analyzer.percentile_values:
        for i in range(len(analyzer.PERCENTILES) + 1):
            if i == 0:
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: Bottom {int(analyzer.PERCENTILES[0]*100)}% (1-{int(analyzer.percentile_values[0])} streak length)"
            elif i == len(analyzer.PERCENTILES):
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: Top {int((1-analyzer.PERCENTILES[-1])*100)}% (>{int(analyzer.percentile_values[-1])} streak length)"
            else:
                lower = int(analyzer.PERCENTILES[i-1]*100)
                upper = int(analyzer.PERCENTILES[i]*100)
                lower_streak = int(analyzer.percentile_values[i-1]) + 1
                upper_streak = int(analyzer.percentile_values[i])
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: {lower}-{upper} percentile ({lower_streak}-{upper_streak} streak length)"

    # Display prediction
    if sorted_pred:
        most_likely_cluster = sorted_pred[0][0]
        most_likely_desc = cluster_descriptions.get(
            most_likely_cluster, f"Cluster {most_likely_cluster}")
        most_likely_prob = float(sorted_pred[0][1]) * 100

        prediction_summary = (
            f"Most likely outcome: {most_likely_desc} ({most_likely_prob:.1f}% probability)\n"
            f"Full probabilities: {prediction}\n"
        )

        # Add next streak number if available
        if recent_streaks and 'streak_number' in recent_streaks[-1]:
            next_streak_num = recent_streaks[-1]['streak_number'] + 1
            prediction_summary += f"Predicting for: Streak #{next_streak_num}"

        print_panel(prediction_summary,
                    title="Streak-Based Prediction Result", style="green")

    print_success("Prediction complete!")


if __name__ == "__main__":
    main()
