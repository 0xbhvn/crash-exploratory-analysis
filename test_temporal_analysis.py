#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for temporal analysis.

This script runs the temporal analysis with various options to test functionality.
"""

import os
import subprocess
import argparse
from logger_config import print_info, print_success, print_error, print_panel, setup_logging
from rich_summary import display_output_summary

# Setup logging
logger = setup_logging()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test Temporal Analysis')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                        help='Mode to run: train a new model or use existing model for prediction')
    parser.add_argument('--input', default='games.csv',
                        help='Input data file path')
    parser.add_argument('--lookback', type=int, default=10,
                        help='Number of previous streaks to use for features')
    parser.add_argument('--multiplier', type=float, default=10.0,
                        help='Multiplier threshold for streak analysis')
    return parser.parse_args()


def run_test(mode, input_file, lookback, multiplier):
    """Run the temporal analysis with specified parameters."""
    print_panel(
        f"Testing Temporal Analysis\nMode: {mode}\nInput: {input_file}\nLookback: {lookback}",
        title="Test Configuration",
        style="blue"
    )

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Base command
    cmd = ["python", "temporal_analysis.py",
           "--mode", mode,
           "--input", input_file,
           "--multiplier_threshold", str(multiplier),
           "--lookback", str(lookback)]

    # Run the command
    print_info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True)
        print_success(
            f"Test completed successfully with exit code: {result.returncode}")

        # Display rich summary of outputs
        print_info("Displaying output summary:")
        display_output_summary("./output")

        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Test failed with exit code: {e.returncode}")
        return False


def main():
    """Main function to run the test."""
    # Parse arguments
    args = parse_arguments()

    # Check if the environment is activated
    if "VIRTUAL_ENV" not in os.environ:
        print_error("Virtual environment is not activated!")
        print_info("Please activate it with: source crash_env/bin/activate")
        return False

    # Check if temporal_analysis.py exists
    if not os.path.exists("temporal_analysis.py"):
        print_error("temporal_analysis.py does not exist!")
        return False

    # Run the test
    success = run_test(args.mode, args.input, args.lookback, args.multiplier)

    return success


if __name__ == "__main__":
    main()
