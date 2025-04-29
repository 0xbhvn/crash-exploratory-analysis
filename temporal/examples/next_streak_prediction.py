#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating how to predict the next streak.

This script shows how to load a trained model and make predictions
for the next streak using the most recent streak data.
"""

from temporal.loader import load_data
from temporal.deploy import load_model_and_predict, setup_prediction_service
import os
import sys
import pandas as pd
import argparse
from utils.logger_config import setup_logging, print_info, print_success, print_panel

# Add parent directory to path to import temporal package
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict the next streak using a trained temporal model')

    parser.add_argument('--model_dir', type=str, default='./output',
                        help='Directory where the model is saved')

    parser.add_argument('--model_file', type=str, default='temporal_model.pkl',
                        help='Filename of the model')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with game data')

    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for streak hits (default: 10.0)')

    parser.add_argument('--streak_count', type=int, default=50,
                        help='Number of most recent streaks to use (default: 50)')

    return parser.parse_args()


def main():
    """Main function to demonstrate next streak prediction."""
    # Set up logging
    setup_logging()

    # Parse arguments
    args = parse_args()

    # Display welcome panel
    print_panel(
        "Next Streak Prediction Example\n"
        f"Using model from {os.path.join(args.model_dir, args.model_file)}\n"
        f"and data from {args.input}",
        title="Temporal Analysis - Forward Prediction",
        style="green bold"
    )

    # Load data
    print_info(f"Loading streak data from {args.input}")
    streak_df = load_data(args.input, args.multiplier_threshold)

    # Use most recent streaks
    if args.streak_count and args.streak_count < len(streak_df):
        recent_streaks = streak_df.tail(args.streak_count).copy()
        print_info(
            f"Using the {args.streak_count} most recent streaks for prediction")
    else:
        recent_streaks = streak_df.copy()
        print_info(f"Using all {len(streak_df)} streaks for prediction")

    # Load model and predict next streak
    model_path = os.path.join(args.model_dir, args.model_file)
    prediction = load_model_and_predict(model_path, recent_streaks)

    # Display success message
    print_success("Next streak prediction complete!")

    # Return prediction for potential further use
    return prediction


if __name__ == "__main__":
    main()
