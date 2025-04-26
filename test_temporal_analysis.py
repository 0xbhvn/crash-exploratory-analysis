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
import unittest
import pandas as pd
import numpy as np
import shutil
from temporal_analysis import create_temporal_features, temporal_train_test_split, preprocess_for_model

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
    parser.add_argument('--lookback', type=int, default=5,
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


class TestTemporalAnalysis(unittest.TestCase):

    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        self.sample_data = pd.DataFrame({
            'id': range(1, 101),
            'timestamp': dates,
            'crash_point': np.random.uniform(1.0, 20.0, 100),
            'streak_length': np.random.randint(1, 15, 100),
            'streak_hit': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
            'temporal_idx': range(100)
        })

        # Create output directory if needed
        if not os.path.exists('test_output'):
            os.makedirs('test_output')

    def tearDown(self):
        # Clean up test files
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')

    def test_create_temporal_features(self):
        # Test with lookback window of 5
        features_df, feature_cols, _ = create_temporal_features(
            self.sample_data, lookback_window=5)

        # Check that we have the expected features
        self.assertGreater(len(feature_cols), 0)

        # Expected number of rows should be original length minus lookback
        expected_rows = len(self.sample_data) - 5
        self.assertEqual(len(features_df), expected_rows)

        # Check for specific feature patterns
        self.assertTrue(
            any('prev_streak_length' in col for col in feature_cols))
        self.assertTrue(any('streak_hit' in col for col in feature_cols))

    def test_temporal_train_test_split(self):
        # Create features first
        features_df, feature_cols, _ = create_temporal_features(
            self.sample_data, lookback_window=5)

        # Test the temporal split with 20% test size
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            features_df, feature_cols, test_size=0.2)

        # Calculate expected sizes
        total_rows = len(features_df)
        expected_test_size = int(0.2 * total_rows)
        expected_train_size = total_rows - expected_test_size

        # Verify the splits have correct sizes
        self.assertEqual(len(X_train), expected_train_size)
        self.assertEqual(len(X_test), expected_test_size)
        self.assertEqual(len(y_train), expected_train_size)
        self.assertEqual(len(y_test), expected_test_size)

        # Verify temporal ordering (train should be before test)
        self.assertTrue(X_train['temporal_idx'].max()
                        < X_test['temporal_idx'].min())

    def test_preprocess_for_model(self):
        # Create features
        features_df, feature_cols, _ = create_temporal_features(
            self.sample_data, lookback_window=5)

        # Test preprocessing
        X, y, scaler = preprocess_for_model(features_df, feature_cols)

        # Check dimensions
        self.assertEqual(X.shape[1], len(feature_cols))
        self.assertEqual(X.shape[0], len(y))

        # Check that scaling was applied (values should be mostly between -5 and 5)
        self.assertTrue(np.all(X <= 5) and np.all(X >= -5))

        # Test scaler
        self.assertIsNotNone(scaler)


if __name__ == "__main__":
    main()
    unittest.main()
