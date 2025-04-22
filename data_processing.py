#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing module for Crash Game 10× Streak Analysis.

This module handles data loading, cleaning, and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

# Import rich logging
from logger_config import (
    console, print_info, print_success, print_warning,
    print_error, create_stats_table
)

logger = logging.getLogger(__name__)


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load and clean crash game data from CSV file.

    Args:
        csv_path: Path to CSV file with game data

    Returns:
        DataFrame with cleaned game data
    """
    logger.info(f"Loading data from {csv_path}")

    df_raw = pd.read_csv(csv_path)
    logger.info(f"Raw data shape: {df_raw.shape}")

    # Clean data
    df = (
        df_raw
        .dropna()
        .rename(columns=lambda c: c.strip())  # trim spaces just in case
    )
    df["Game ID"] = df["Game ID"].astype(int)
    df["Bust"] = df["Bust"].astype(float)

    # Log cleaning operations
    cleaned_stats = {
        "Raw Shape": str(df_raw.shape),
        "Cleaned Shape": str(df.shape),
        "Rows Removed": df_raw.shape[0] - df.shape[0],
        "Columns": ", ".join(df.columns.tolist())
    }
    create_stats_table("Data Cleaning", cleaned_stats)

    return df


def analyze_streaks(df: pd.DataFrame, multiplier_threshold: float = 10.0) -> List[int]:
    """
    Analyze streak lengths including multipliers at or above the threshold.

    Args:
        df: DataFrame with game data
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)

    Returns:
        List of streak lengths (including the game with ≥threshold multiplier)
    """
    logger.info(
        f"Analyzing streak lengths including {multiplier_threshold}× multipliers")

    streak_lengths = []
    current_streak_length = 0

    for bust in df["Bust"]:
        # Increment streak for all games (including ≥threshold)
        current_streak_length += 1

        if bust >= multiplier_threshold:
            # We hit ≥threshold ⇒ record the streak length (including this game)
            streak_lengths.append(current_streak_length)
            current_streak_length = 0  # reset

    # If the dataset ends without a threshold hit, we drop the trailing incomplete streak

    # Display streak statistics
    streaks_stats = {
        f"Total {multiplier_threshold}× Hits": len(streak_lengths),
        "Min Streak Length": min(streak_lengths) if streak_lengths else "N/A",
        "Max Streak Length": max(streak_lengths) if streak_lengths else "N/A",
        "Avg Streak Length": f"{np.mean(streak_lengths):.2f}" if streak_lengths else "N/A"
    }
    create_stats_table("Streak Analysis", streaks_stats)

    return streak_lengths


def calculate_streak_percentiles(streak_lengths: List[int]) -> Dict[str, float]:
    """
    Calculate percentiles for streak lengths.

    Args:
        streak_lengths: List of streak lengths

    Returns:
        Dictionary of percentiles
    """
    percentiles = {
        "P50": np.percentile(streak_lengths, 50),
        "P75": np.percentile(streak_lengths, 75),
        "P90": np.percentile(streak_lengths, 90),
        "P95": np.percentile(streak_lengths, 95),
        "P99": np.percentile(streak_lengths, 99),
    }
    logger.info(f"Streak length percentiles: {percentiles}")
    return percentiles


def add_rolling_features(frame: pd.DataFrame, window: int, multiplier_threshold: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add rolling window features to DataFrame.

    Args:
        frame: DataFrame with game data
        window: Rolling window size
        multiplier_threshold: Threshold for considering a multiplier as a hit

    Returns:
        Tuple containing processed DataFrame and list of feature column names
    """
    print_info(f"Adding rolling window features (window={window})")

    roll = frame["Bust"].rolling(window, min_periods=1)
    frame[f"mean_{window}"] = roll.mean()
    frame[f"std_{window}"] = roll.std().fillna(0)
    frame[f"max_{window}"] = roll.max()
    frame[f"min_{window}"] = roll.min()
    frame[f"pct_gt2_{window}"] = frame["Bust"].gt(2).rolling(window).mean()
    frame[f"pct_gt5_{window}"] = frame["Bust"].gt(5).rolling(window).mean()

    print_info(f"Added 6 rolling features: mean, std, max, min, pct_gt2, pct_gt5")

    # Define feature columns
    feature_cols = [
        # Rolling statistics features
        f"mean_{window}",
        f"std_{window}",
        f"max_{window}",
        f"min_{window}",
        f"pct_gt2_{window}",
        f"pct_gt5_{window}"
    ]

    # Add interaction features if all components are present
    interaction_features = [
        (f"mean_{window}", f"pct_gt2_{window}"),
        (f"mean_{window}", f"pct_gt5_{window}"),
        (f"mean_{window}", f"std_{window}")
    ]

    for feat1, feat2 in interaction_features:
        if feat1 in frame.columns and feat2 in frame.columns:
            feat_name = f"{feat1}_x_{feat2}"
            frame[feat_name] = frame[feat1] * frame[feat2]
            feature_cols.append(feat_name)

    # Feature scaling
    scaler = StandardScaler()
    frame[feature_cols] = scaler.fit_transform(frame[feature_cols])

    # Summary information
    logger.info(f"Prepared {len(feature_cols)} features for machine learning")
    logger.info(f"Final feature matrix shape: {frame.shape}")

    return frame, feature_cols


def prepare_features(df: pd.DataFrame,
                     window: int,
                     multiplier_threshold: float = 10.0,
                     percentiles: List[float] = [0.25, 0.50, 0.75]
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for machine learning model training.

    This function creates rolling window features and labels each target based on the
    length of the next streak that includes a hit at or above the multiplier threshold.
    The clustering is done using customizable percentiles of streak lengths.

    Args:
        df: DataFrame with game data
        window: Rolling window size for feature engineering
        multiplier_threshold: Threshold for considering a multiplier as a hit
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])

    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    logger.info(f"Preparing features with window={window}")

    # Feature engineering - Add rolling window features
    df, rolling_feat_cols = add_rolling_features(
        df, window, multiplier_threshold)

    feature_cols = rolling_feat_cols.copy()
    logger.debug(f"Created {len(feature_cols)} features")

    # Add hit column for the multiplier threshold
    # Convert threshold to string (integer if whole number)
    threshold_str = str(int(multiplier_threshold)
                        if multiplier_threshold.is_integer() else multiplier_threshold)
    hit_col = f"is_hit{threshold_str}"
    df[hit_col] = (df['Bust'] >= multiplier_threshold).astype(int)
    logger.info(
        f"Added column {hit_col} to track hits at or above {multiplier_threshold}×")

    # Create target variable - Get the streak length that includes the next hit
    # More efficient algorithm to calculate next streak lengths
    print_info("Calculating next streak lengths (this may take a moment)...")

    # Find all hit positions
    hit_positions = np.where(df['Bust'].values >= multiplier_threshold)[0]

    # Create a mapping for each position to the next hit position
    next_hit_map = {}
    for i in range(len(hit_positions) - 1):
        current_pos = hit_positions[i]
        next_pos = hit_positions[i + 1]
        # For all positions between current and next hit, the next hit is at next_pos
        for pos in range(current_pos + 1, next_pos + 1):
            next_hit_map[pos] = next_pos

    # Calculate next streak lengths for the window positions
    next_streak_lengths = []
    max_valid_idx = len(df) - 1

    for i in range(len(df) - window):
        start_idx = i + window
        if start_idx in next_hit_map and start_idx <= max_valid_idx:
            # Calculate streak length (add 1 because we include the hit game)
            streak_length = next_hit_map[start_idx] - start_idx + 1
            next_streak_lengths.append(streak_length)
        else:
            next_streak_lengths.append(np.nan)

    # Create a new DataFrame with features and target, dropping last window rows
    features_df = df.iloc[:-window].copy()
    features_df['next_streak_length'] = next_streak_lengths

    # Remove rows with NaN target values
    features_df = features_df.dropna(subset=['next_streak_length'])
    logger.info(f"DataFrame after dropping NaN targets: {features_df.shape}")

    # Calculate percentile boundaries for clustering
    percentile_values = [
        features_df['next_streak_length'].quantile(p) for p in percentiles]

    # Log percentile boundaries
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val:.1f}" for p, val in zip(percentiles, percentile_values)])
    logger.info(f"Streak length percentile boundaries: {percentile_info}")

    # Create conditions for clustering based on percentile boundaries
    conditions = []
    for i in range(len(percentile_values) + 1):
        if i == 0:
            # First cluster: <= first percentile
            conditions.append(
                features_df['next_streak_length'] <= percentile_values[0])
        elif i == len(percentile_values):
            # Last cluster: > last percentile
            conditions.append(
                features_df['next_streak_length'] > percentile_values[-1])
        else:
            # Middle clusters: between adjacent percentiles
            conditions.append(
                (features_df['next_streak_length'] > percentile_values[i-1]) &
                (features_df['next_streak_length'] <= percentile_values[i])
            )

    # Create clusters (0 to n, where n is the number of percentile boundaries + 1)
    clusters = list(range(len(percentile_values) + 1))
    features_df['target_cluster'] = np.select(
        conditions, clusters, default=np.nan)

    # Log cluster counts
    cluster_counts = []
    for i in clusters:
        count = (features_df['target_cluster'] == i).sum()
        percentage = count / len(features_df) * 100
        cluster_counts.append(f"{i}: {count} ({percentage:.1f}%)")

    logger.info(
        f"Percentile-based cluster counts: {', '.join(cluster_counts)}")
    logger.info(
        f"Final feature matrix shape: {features_df[feature_cols].shape}")

    return features_df, feature_cols


def make_feature_vector(last_window_multipliers: List[float], window: int, feature_cols: List[str]) -> pd.Series:
    """
    Create a feature vector from recent multipliers.

    Args:
        last_window_multipliers: List of window most recent multipliers
        window: Rolling window size
        feature_cols: List of feature column names to include

    Returns:
        Series with feature values
    """
    seq = pd.Series(last_window_multipliers)

    feat = {}
    # Add rolling features
    feat[f"mean_{window}"] = seq.mean()
    feat[f"std_{window}"] = seq.std()
    feat[f"max_{window}"] = seq.max()
    feat[f"min_{window}"] = seq.min()
    feat[f"pct_gt2_{window}"] = seq.gt(2).mean()
    feat[f"pct_gt5_{window}"] = seq.gt(5).mean()

    # Add lagged values
    for i, x in enumerate(reversed(last_window_multipliers), 1):
        if i <= window:
            feat[f"m_{i}"] = x

    # Create Series with same names as training data
    result = pd.Series({k: feat.get(k, np.nan) for k in feature_cols})
    return result
