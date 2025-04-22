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


def add_rolling_features(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Add rolling window features to the dataframe.

    Args:
        frame: DataFrame to add features to
        window: Rolling window size

    Returns:
        DataFrame with added features
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

    return frame


def prepare_features(df: pd.DataFrame, window: int, clusters: Dict[int, Tuple[int, int]], multiplier_threshold: float = 10.0) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for machine learning, including:
    - Mark multiplier threshold hits
    - Calculate distance to next threshold hit (including the hit)
    - Create cluster labels
    - Generate rolling window features

    Args:
        df: DataFrame with game data
        window: Rolling window size for feature engineering
        clusters: Dictionary mapping cluster IDs to (min, max) streak length ranges
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)

    Returns:
        Tuple of (DataFrame with features and target, list of feature column names)
    """
    logger.info("Preparing features for machine learning")

    # Sort by Game ID to ensure chronological order
    df = df.sort_values("Game ID").reset_index(drop=True)

    # Mark threshold hits
    hit_col = f"is_hit{int(multiplier_threshold) if multiplier_threshold.is_integer() else multiplier_threshold}"
    df[hit_col] = (df["Bust"] >= multiplier_threshold).astype(int)
    print_info(
        f"Marked {df[hit_col].sum()} {multiplier_threshold}× hits out of {len(df)} games")

    # Distance to next threshold hit (including the hit game itself)
    n = len(df)
    gap = np.empty(n, dtype=int)
    dist = n  # start with a large but finite number

    for i in range(n - 1, -1, -1):
        dist += 1  # increment distance for all games
        if df.at[i, hit_col]:
            gap[i] = 0  # this game is a threshold hit
            dist = 0  # reset counter from this game
        else:
            gap[i] = dist  # distance to next threshold hit

    df["gap_next_hit"] = gap
    print_info(
        f"Calculated distance to next {multiplier_threshold}× for each game")

    # Map gap to cluster
    def gap_to_cluster(g):
        # Include the game with the threshold hit in the streak length
        streak_length = g + 1  # +1 to include the hit game

        # If this is a threshold hit game itself, set streak_length to 1
        if g == 0:
            streak_length = 1

        for c, (lo, hi) in clusters.items():
            if lo <= streak_length <= hi:
                return c
        return np.nan  # anything >9999 → NaN

    df["target_cluster"] = df["gap_next_hit"].map(gap_to_cluster)
    df = df.dropna(subset=["target_cluster"]).reset_index(drop=True)
    df["target_cluster"] = df["target_cluster"].astype(int)

    # Display cluster statistics
    cluster_stats = {}
    for cluster_id in range(len(clusters)):
        count = (df["target_cluster"] == cluster_id).sum()
        percentage = count / len(df) * 100
        cluster_stats[f"Cluster {cluster_id}"] = f"{count} games ({percentage:.1f}%)"

    create_stats_table("Target Cluster Distribution", cluster_stats)

    # Add rolling window features
    df = add_rolling_features(df, window)

    # Add lag features
    print_info(f"Adding {window} lag features")
    for lag in range(1, window + 1):
        df[f"m_{lag}"] = df["Bust"].shift(lag)

    # Drop rows with NaNs (from shifting)
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    print_info(f"Dropped {rows_before - rows_after} rows with NaN values")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in
                    ("Game ID", hit_col, "gap_next_hit", "target_cluster")]

    feature_stats = {
        "Total Features": len(feature_cols),
        "Rolling Features": sum(1 for c in feature_cols if c.startswith(("mean_", "std_", "max_", "min_", "pct_"))),
        "Lag Features": sum(1 for c in feature_cols if c.startswith("m_")),
        "Final Matrix Shape": f"{df[feature_cols].shape[0]} rows × {df[feature_cols].shape[1]} columns"
    }
    create_stats_table("Feature Engineering Summary", feature_stats)

    print_success("Feature preparation complete")

    return df, feature_cols


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
