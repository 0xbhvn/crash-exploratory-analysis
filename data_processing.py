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

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def analyze_streaks(df: pd.DataFrame) -> List[int]:
    """
    Analyze streak lengths before 10× multipliers.

    Args:
        df: DataFrame with game data

    Returns:
        List of streak lengths
    """
    logger.info("Analyzing streak lengths before 10× multipliers")

    streak_lengths = []
    current_streak_length = 0

    for bust in df["Bust"]:
        if bust < 10:
            current_streak_length += 1
        else:
            # We hit ≥10×  ⇒ record the *preceding* streak length
            streak_lengths.append(current_streak_length)
            current_streak_length = 0  # reset

    # If the dataset ends without a 10×, we drop the trailing incomplete streak
    streak_lengths = [s for s in streak_lengths if s > 0]

    logger.info(f"Collected {len(streak_lengths):,} streaks")

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
    roll = frame["Bust"].rolling(window, min_periods=1)
    frame[f"mean_{window}"] = roll.mean()
    frame[f"std_{window}"] = roll.std().fillna(0)
    frame[f"max_{window}"] = roll.max()
    frame[f"min_{window}"] = roll.min()
    frame[f"pct_gt2_{window}"] = frame["Bust"].gt(2).rolling(window).mean()
    frame[f"pct_gt5_{window}"] = frame["Bust"].gt(5).rolling(window).mean()
    return frame


def prepare_features(df: pd.DataFrame, window: int, clusters: Dict[int, Tuple[int, int]]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for machine learning, including:
    - Mark 10× hits
    - Calculate distance to next 10× hit
    - Create cluster labels
    - Generate rolling window features

    Args:
        df: DataFrame with game data
        window: Rolling window size for feature engineering
        clusters: Dictionary mapping cluster IDs to (min, max) streak length ranges

    Returns:
        Tuple of (DataFrame with features and target, list of feature column names)
    """
    logger.info("Preparing features for machine learning")

    # Sort by Game ID to ensure chronological order
    df = df.sort_values("Game ID").reset_index(drop=True)

    # Mark 10× hits
    df["is_hit10"] = (df["Bust"] >= 10).astype(int)

    # Distance to next 10×
    n = len(df)
    gap = np.empty(n, dtype=int)
    dist = n  # start with a large but finite number

    for i in range(n - 1, -1, -1):
        if df.at[i, "is_hit10"]:
            dist = 0  # reset on every hit
        else:
            dist += 1  # count how far we are
        gap[i] = dist

    df["gap_next_10x"] = gap

    # Map gap to cluster
    def gap_to_cluster(g):
        for c, (lo, hi) in clusters.items():
            if lo <= g <= hi:
                return c
        return np.nan  # anything >9999 → NaN

    df["target_cluster"] = df["gap_next_10x"].map(gap_to_cluster)
    df = df.dropna(subset=["target_cluster"]).reset_index(drop=True)
    df["target_cluster"] = df["target_cluster"].astype(int)

    logger.info(f"Rows with cluster label: {df.target_cluster.notna().sum()}")

    # Add rolling window features
    df = add_rolling_features(df, window)

    # Add lag features
    for lag in range(1, window + 1):
        df[f"m_{lag}"] = df["Bust"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in
                    ("Game ID", "is_hit10", "gap_next_10x", "target_cluster")]

    logger.info(f"Final feature matrix shape: {df[feature_cols].shape}")
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
    feat["Bust"] = seq.iloc[-1]
    feat[f"mean_{window}"] = seq.mean()
    feat[f"std_{window}"] = seq.std()
    feat[f"max_{window}"] = seq.max()
    feat[f"min_{window}"] = seq.min()
    feat[f"pct_gt2_{window}"] = (seq > 2).mean()
    feat[f"pct_gt5_{window}"] = (seq > 5).mean()

    for lag, val in enumerate(last_window_multipliers, 1):
        feat[f"m_{lag}"] = val

    # Ensure correct order
    return pd.Series(feat).reindex(feature_cols)
