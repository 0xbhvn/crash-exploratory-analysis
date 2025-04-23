#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyzer class for Crash Game Streak Analysis.

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

# Import rich logging
from logger_config import (
    console, create_table, display_table, add_table_row,
    create_stats_table, print_info, print_success, print_warning,
    print_error, print_panel
)

logger = logging.getLogger(__name__)


class CrashStreakAnalyzer:
    """
    Analyzes crash game data to predict streak lengths including configurable multipliers.

    This class handles data loading, cleaning, feature engineering, model training,
    and prediction of streak length categories. Streak lengths include all games
    up to and including the specified multiplier threshold.
    """

    def __init__(self,
                 multiplier_threshold: float = 10.0,
                 window: int = 50,
                 test_frac: float = 0.2,
                 random_seed: int = 42,
                 output_dir: str = './output',
                 percentiles: List[float] = [0.25, 0.50, 0.75]):
        """
        Initialize the analyzer with configuration parameters.

        Args:
            multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
            window: Rolling window size for feature engineering
            test_frac: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
            output_dir: Directory to save outputs
            percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])
        """
        self.MULTIPLIER_THRESHOLD = multiplier_threshold
        self.WINDOW = window
        self.TEST_FRAC = test_frac
        self.RANDOM_SEED = random_seed
        self.output_dir = output_dir
        self.PERCENTILES = percentiles
        self.NUM_CLUSTERS = len(percentiles) + 1

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print_info(f"Created output directory: {output_dir}")

        # Initialize attributes that will be populated later
        self.df = None
        self.feature_cols = None
        self.bst_final = None
        self.p_hat = None
        self.baseline_probs = None
        self.percentile_values = None

        # Display initialization info in a panel
        config_info = (
            f"Multiplier Threshold: {multiplier_threshold}×\n"
            f"Window Size: {window}\n"
            f"Test Fraction: {test_frac}\n"
            f"Random Seed: {random_seed}\n"
            f"Output Directory: {output_dir}\n"
            f"Percentiles: {percentiles}"
        )
        print_panel(config_info, title="Analyzer Configuration", style="blue")

        # Display information about percentile-based clusters
        cluster_table = create_table("Percentile-Based Clustering", [
                                     "Cluster ID", "Description"])

        # Create cluster descriptions based on percentiles
        cluster_descriptions = []
        for i in range(len(percentiles) + 1):
            if i == 0:
                description = f"Cluster {i}: Bottom {int(percentiles[0]*100)}% (shortest streaks)"
            elif i == len(percentiles):
                description = f"Cluster {i}: Top {int((1-percentiles[-1])*100)}% (longest streaks)"
            else:
                lower = int(percentiles[i-1]*100)
                upper = int(percentiles[i]*100)
                description = f"Cluster {i}: {lower}-{upper} percentile"

            cluster_descriptions.append([str(i), description])

        for cluster_id, description in cluster_descriptions:
            add_table_row(cluster_table, [cluster_id, description])
        display_table(cluster_table)

        print_info(
            "Streak length clusters will be determined dynamically based on provided percentiles")

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load and clean crash game data from CSV file.

        Args:
            csv_path: Path to CSV file with game data

        Returns:
            DataFrame with cleaned game data
        """
        print_info(f"Loading data from {csv_path}")
        self.df = data_processing.load_data(csv_path)

        # Display data summary
        summary_stats = {
            "Total Games": len(self.df),
            "First Game ID": self.df["Game ID"].min() if not self.df.empty else "N/A",
            "Last Game ID": self.df["Game ID"].max() if not self.df.empty else "N/A",
            "Min Multiplier": self.df["Bust"].min() if not self.df.empty else "N/A",
            "Max Multiplier": self.df["Bust"].max() if not self.df.empty else "N/A",
            "Avg Multiplier": round(self.df["Bust"].mean(), 2) if not self.df.empty else "N/A",
            f"{self.MULTIPLIER_THRESHOLD}× or Higher": (self.df["Bust"] >= self.MULTIPLIER_THRESHOLD).sum() if not self.df.empty else "N/A",
            f"{self.MULTIPLIER_THRESHOLD}× Rate": f"{(self.df['Bust'] >= self.MULTIPLIER_THRESHOLD).mean() * 100:.2f}%" if not self.df.empty else "N/A"
        }
        create_stats_table("Data Summary", summary_stats)

        return self.df

    def analyze_streaks(self, save_streak_lengths: bool = True) -> List[int]:
        """
        Analyze streak lengths including multipliers at or above the threshold.

        Args:
            save_streak_lengths: Whether to save streak lengths to CSV

        Returns:
            List of streak lengths (including the game with ≥threshold multiplier)
        """
        print_info(
            f"Analyzing streak lengths including {self.MULTIPLIER_THRESHOLD}× multipliers")
        streak_lengths = data_processing.analyze_streaks(
            self.df, self.MULTIPLIER_THRESHOLD)

        # Calculate percentiles
        percentiles = data_processing.calculate_streak_percentiles(
            streak_lengths)

        # Display percentiles in a table
        percentiles_table = create_table(
            "Streak Length Percentiles", ["Percentile", "Value"])
        for percentile, value in percentiles.items():
            add_table_row(percentiles_table, [percentile, f"{value:.1f}"])
        display_table(percentiles_table)

        # Display streak distribution
        freq_table = create_table("Streak Length Frequency", [
                                  "Streak Length", "Frequency", "Percentage"])
        streak_series = pd.Series(streak_lengths)
        value_counts = streak_series.value_counts().sort_index()
        total_streaks = len(streak_lengths)

        # Only display up to 20 rows for readability
        for length, count in value_counts.head(20).items():
            percentage = (count / total_streaks) * 100
            add_table_row(freq_table, [length, count, f"{percentage:.2f}%"])

        if len(value_counts) > 20:
            add_table_row(freq_table, ["...", "...", "..."])

        display_table(freq_table)

        # Save streak lengths if requested
        if save_streak_lengths:
            streak_file = os.path.join(self.output_dir, "streak_lengths.csv")
            pd.Series(streak_lengths, name="streak_length").to_csv(
                streak_file, index=False)
            print_info(f"Saved streak lengths to {streak_file}")

        return streak_lengths

    def plot_streaks(self, streak_lengths: List[int]) -> None:
        """
        Generate and save plots of streak length distributions.

        Args:
            streak_lengths: List of streak lengths
        """
        print_info("Generating streak length distribution plots")
        percentiles = data_processing.calculate_streak_percentiles(
            streak_lengths)
        visualization.plot_streaks(
            streak_lengths, percentiles, self.output_dir, self.MULTIPLIER_THRESHOLD)
        print_success("Saved streak distribution plots to output directory")

    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for machine learning using streak-based analysis.

        Returns:
            DataFrame with features and target clustered by percentiles
        """
        print_info("Preparing streak-based features for machine learning")

        # The window parameter is now used as a lookback window of previous streaks rather than games
        self.df, self.feature_cols = data_processing.prepare_features(
            self.df, self.WINDOW,
            multiplier_threshold=self.MULTIPLIER_THRESHOLD,
            percentiles=self.PERCENTILES)

        # Get the percentile values for display purposes
        if 'target_streak_length' in self.df.columns:
            self.percentile_values = [
                self.df['target_streak_length'].quantile(p) for p in self.PERCENTILES]

        # Display feature information
        feature_info = {
            "Number of Features": len(self.feature_cols),
            "Streaks Analyzed": len(self.df),
            "Target Classes": len(self.df["target_cluster"].unique()),
        }

        # Add class counts to feature_info
        for i in range(self.NUM_CLUSTERS):
            if i == 0:
                description = f"Class {i} (Bottom {int(self.PERCENTILES[0]*100)}%, 1-{int(self.percentile_values[0])} streak length)"
            elif i == self.NUM_CLUSTERS - 1:
                description = f"Class {i} (Top {int((1-self.PERCENTILES[-1])*100)}%, >{int(self.percentile_values[-1])} streak length)"
            else:
                lower = int(self.PERCENTILES[i-1]*100)
                upper = int(self.PERCENTILES[i]*100)
                lower_streak = int(self.percentile_values[i-1]) + 1
                upper_streak = int(self.percentile_values[i])
                description = f"Class {i} ({lower}-{upper}%, {lower_streak}-{upper_streak} streak length)"

            feature_info[description] = (self.df["target_cluster"] == i).sum()

        create_stats_table("Streak Feature Matrix Information", feature_info)

        # Display sample of features
        if len(self.feature_cols) > 0:
            feature_sample = self.df[self.feature_cols +
                                     ["target_cluster"]].head(5)

            # Create a table with just a few columns for readability
            # Show only first 5 features
            cols_to_show = self.feature_cols[:5] + ["target_cluster"]

            sample_table = create_table(
                "Streak Feature Sample (First 5 columns)", cols_to_show)
            for _, row in feature_sample.iterrows():
                row_values = [f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
                              for col in cols_to_show]
                add_table_row(sample_table, row_values)

            display_table(sample_table)

            if len(self.feature_cols) > 5:
                print_info(
                    f"Showing 5 of {len(self.feature_cols)} features. Full matrix: {self.df[self.feature_cols].shape}")

        return self.df

    def train_model(self, eval_folds: int = 5) -> xgb.Booster:
        """
        Train a model to predict streak length clusters.

        Args:
            eval_folds: Number of folds for rolling-origin cross-validation

        Returns:
            Trained XGBoost model
        """
        print_info("Training model to predict streak length clusters")

        model_results = modeling.train_model(
            self.df, self.feature_cols,
            self.TEST_FRAC, self.RANDOM_SEED, eval_folds,
            self.output_dir, self.MULTIPLIER_THRESHOLD,
            percentiles=self.PERCENTILES,
            window=self.WINDOW
        )

        # model_results can be either a tuple of (model, baseline_probs, p_hat)
        # or a new-style bundle including the model
        if isinstance(model_results, tuple) and len(model_results) == 3:
            self.bst_final, self.baseline_probs, self.p_hat = model_results
        else:
            # Store the entire bundle
            self.bst_final = model_results
            if isinstance(model_results, dict):
                if "model" in model_results and hasattr(model_results["model"], "predict"):
                    # Extract baseline_probs and p_hat if they're in the bundle
                    self.baseline_probs = model_results.get(
                        "baseline_probs", {})
                    self.p_hat = model_results.get("p_hat", 1.0)

        # Generate feature importance plot
        model_to_plot = self.bst_final
        if isinstance(self.bst_final, dict) and "model" in self.bst_final:
            model_to_plot = self.bst_final["model"]

        visualization.plot_feature_importance(
            model_to_plot, self.feature_cols, self.output_dir)

        print_success("Model training complete")

        # Display model evaluation metrics if available
        if hasattr(model_to_plot, 'best_score'):
            metrics = {
                "Best Validation Score": model_to_plot.best_score,
                "Number of Trees": model_to_plot.best_iteration + 1,
                f"{self.MULTIPLIER_THRESHOLD}× Base Rate": f"{self.p_hat * 100:.2f}%"
            }

            # Add baseline probabilities
            for cluster_id, prob in self.baseline_probs.items():
                # Create cluster label based on percentiles
                if cluster_id == 0:
                    cluster_name = f"Bottom {int(self.PERCENTILES[0]*100)}%"
                elif cluster_id == len(self.PERCENTILES):
                    cluster_name = f"Top {int((1-self.PERCENTILES[-1])*100)}%"
                else:
                    lower = int(self.PERCENTILES[cluster_id-1]*100)
                    upper = int(self.PERCENTILES[cluster_id]*100)
                    cluster_name = f"{lower}-{upper}%"

                metrics[f"Baseline Prob. ({cluster_name})"] = f"{prob * 100:.2f}%"

            create_stats_table("Model Evaluation", metrics)

        return self.bst_final

    def predict_next_cluster(self, recent_streaks: List[Dict] = None) -> Dict[str, float]:
        """
        Predict the next streak length cluster based on recent streak patterns.

        Args:
            recent_streaks: List of dictionaries with recent streak information.
                           If None, the last self.WINDOW streaks from the dataset will be used.

        Returns:
            Dictionary of cluster probabilities for streak lengths
        """
        if self.bst_final is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # If no streaks provided, extract the last WINDOW streaks from our data
        if recent_streaks is None:
            # Check if we already have a streak DataFrame
            if hasattr(self, 'streak_df') and self.streak_df is not None and not self.streak_df.empty:
                # Use existing streaks
                recent_streaks = self.streak_df.tail(
                    self.WINDOW).to_dict('records')
                print_info(
                    f"Using the last {len(recent_streaks)} streaks from existing streak dataframe for prediction")
            else:
                # Extract streaks from raw data if needed
                from data_processing import extract_streaks_and_multipliers
                try:
                    # Make sure we have the expected columns
                    if "Game ID" in self.df.columns and "Bust" in self.df.columns:
                        self.streak_df = extract_streaks_and_multipliers(
                            self.df, self.MULTIPLIER_THRESHOLD)
                        # Store for future use
                        recent_streaks = self.streak_df.tail(
                            self.WINDOW).to_dict('records')
                        print_info(
                            f"Extracted {len(self.streak_df)} streaks, using last {len(recent_streaks)} for prediction")
                    else:
                        # Handle the case where self.df has already been processed
                        print_warning(
                            "DataFrame doesn't have expected columns. Creating features directly.")
                        if 'target_cluster' in self.df.columns:
                            # Already processed, reuse it directly
                            recent_streaks = self.df.head(
                                self.WINDOW).to_dict('records')
                        else:
                            raise ValueError(
                                "Cannot extract streaks - DataFrame format not recognized")
                except Exception as e:
                    print_error(f"Error extracting streaks: {str(e)}")
                    # Create empty streaks as fallback
                    recent_streaks = []

        # Check if bst_final is a model or a bundle
        model = self.bst_final
        feature_cols = self.feature_cols

        # For backward compatibility: if bst_final is a dict/bundle, extract model and feature_cols
        if isinstance(self.bst_final, dict) and "model" in self.bst_final:
            model = self.bst_final["model"]
            if "feature_cols" in self.bst_final:
                feature_cols = self.bst_final["feature_cols"]

        results = modeling.predict_next_cluster(
            model, recent_streaks,
            self.WINDOW, feature_cols, self.MULTIPLIER_THRESHOLD,
            percentiles=self.PERCENTILES
        )

        # Display prediction results in a table
        prediction_table = create_table(
            "Streak Prediction Results", ["Cluster", "Description", "Probability"])

        # Generate cluster descriptions based on percentiles
        cluster_descriptions = {}
        for i in range(self.NUM_CLUSTERS):
            if i == 0:
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: Bottom {int(self.PERCENTILES[0]*100)}% (1-{int(self.percentile_values[0])} streak length)"
            elif i == self.NUM_CLUSTERS - 1:
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: Top {int((1-self.PERCENTILES[-1])*100)}% (>{int(self.percentile_values[-1])} streak length)"
            else:
                lower = int(self.PERCENTILES[i-1]*100)
                upper = int(self.PERCENTILES[i]*100)
                lower_streak = int(self.percentile_values[i-1]) + 1
                upper_streak = int(self.percentile_values[i])
                cluster_descriptions[str(
                    i)] = f"Cluster {i}: {lower}-{upper} percentile ({lower_streak}-{upper_streak} streak length)"

        # Sort by probability (descending)
        sorted_results = sorted(
            results.items(), key=lambda x: float(x[1]), reverse=True)

        for cluster, prob in sorted_results:
            description = cluster_descriptions.get(cluster, "Unknown")
            add_table_row(prediction_table, [
                          cluster, description, f"{float(prob) * 100:.2f}%"])

        display_table(prediction_table)

        return results

    def daily_update(self, new_rows: pd.DataFrame, drift_threshold: float = 0.005) -> bool:
        """
        Update model with new data and potentially retrain if drift is detected.

        Args:
            new_rows: DataFrame with new game data
            drift_threshold: Threshold for detecting drift in multiplier rate

        Returns:
            Boolean indicating whether model was retrained
        """
        print_info("Processing daily update")
        from daily_updates import process_daily_update
        return process_daily_update(self, new_rows, drift_threshold)

    def save_snapshot(self) -> None:
        """
        Save a snapshot of the current model and data.
        """
        print_info("Saving model and data snapshot")
        from daily_updates import save_model_snapshot
        save_model_snapshot(self)
        print_success("Model snapshot saved successfully")
