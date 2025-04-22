#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyzer class for Crash Game 10× Streak Analysis.

This module contains the main CrashStreakAnalyzer class that coordinates the analysis.
"""

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Any

# Import from local modules
import data_processing
import modeling
import visualization

logger = logging.getLogger(__name__)


class CrashStreakAnalyzer:
    """
    Analyzes crash game data to predict streak lengths before 10× multipliers.

    This class handles data loading, cleaning, feature engineering, model training,
    and prediction of streak length categories.
    """

    def __init__(self,
                 window: int = 50,
                 clusters: Dict[int, Tuple[int, int]] = None,
                 test_frac: float = 0.2,
                 random_seed: int = 42,
                 output_dir: str = './output'):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            window: Rolling window size for feature engineering
            clusters: Dictionary mapping cluster IDs to (min, max) streak length ranges
            test_frac: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
            output_dir: Directory to save outputs
        """
        self.WINDOW = window
        self.CLUSTERS = clusters or {0: (1, 5), 1: (6, 12), 2: (13, 9999)}
        self.TEST_FRAC = test_frac
        self.RANDOM_SEED = random_seed
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

        # Initialize attributes that will be populated later
        self.df = None
        self.feature_cols = None
        self.bst_final = None
        self.p_hat = None
        self.baseline_probs = None

        logger.info(f"Initialized CrashStreakAnalyzer with window={window}, "
                    f"test_frac={test_frac}, random_seed={random_seed}")

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and clean crash game data from CSV file.

        Args:
            csv_path: Path to CSV file with game data

        Returns:
            DataFrame with cleaned game data
        """
        self.df = data_processing.load_data(csv_path)
        return self.df

    def analyze_streaks(self, save_streak_lengths: bool = True) -> List[int]:
        """
        Analyze streak lengths before 10× multipliers.

        Args:
            save_streak_lengths: Whether to save streak lengths to CSV

        Returns:
            List of streak lengths
        """
        streak_lengths = data_processing.analyze_streaks(self.df)

        # Calculate percentiles
        percentiles = data_processing.calculate_streak_percentiles(
            streak_lengths)

        # Save streak lengths if requested
        if save_streak_lengths:
            streak_file = os.path.join(self.output_dir, "streak_lengths.csv")
            pd.Series(streak_lengths, name="streak_length").to_csv(
                streak_file, index=False)
            logger.info(f"Saved streak lengths to {streak_file}")

        return streak_lengths

    def plot_streaks(self, streak_lengths: List[int]) -> None:
        """
        Generate and save plots of streak length distributions.

        Args:
            streak_lengths: List of streak lengths
        """
        percentiles = data_processing.calculate_streak_percentiles(
            streak_lengths)
        visualization.plot_streaks(
            streak_lengths, percentiles, self.output_dir)

    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for machine learning.

        Returns:
            DataFrame with features and target
        """
        self.df, self.feature_cols = data_processing.prepare_features(
            self.df, self.WINDOW, self.CLUSTERS)
        return self.df

    def train_model(self, eval_folds: int = 5) -> xgb.Booster:
        """
        Train a model to predict streak length clusters.

        Args:
            eval_folds: Number of folds for rolling-origin cross-validation

        Returns:
            Trained XGBoost model
        """
        self.bst_final, self.baseline_probs, self.p_hat = modeling.train_model(
            self.df, self.feature_cols, self.CLUSTERS,
            self.TEST_FRAC, self.RANDOM_SEED, eval_folds,
            self.output_dir
        )

        # Generate feature importance plot
        visualization.plot_feature_importance(
            self.bst_final, self.feature_cols, self.output_dir)

        return self.bst_final

    def predict_next_cluster(self, last_window_multipliers: List[float]) -> Dict[str, float]:
        """
        Predict the next cluster based on recent multipliers.

        Args:
            last_window_multipliers: List of last WINDOW multipliers

        Returns:
            Dictionary of cluster probabilities
        """
        if self.bst_final is None:
            raise ValueError("Model not trained. Call train_model() first.")

        return modeling.predict_next_cluster(
            self.bst_final, last_window_multipliers,
            self.WINDOW, self.feature_cols
        )

    def daily_update(self, new_rows: pd.DataFrame, drift_threshold: float = 0.005) -> bool:
        """
        Update model with new data and potentially retrain if drift is detected.

        Args:
            new_rows: DataFrame with new game data
            drift_threshold: Threshold for detecting drift in 10× rate

        Returns:
            Boolean indicating whether model was retrained
        """
        from daily_updates import process_daily_update
        return process_daily_update(self, new_rows, drift_threshold)

    def save_snapshot(self) -> None:
        """
        Save a snapshot of the current model and data.
        """
        from daily_updates import save_model_snapshot
        save_model_snapshot(self)
