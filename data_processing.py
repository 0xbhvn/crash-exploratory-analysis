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
        "P25": np.percentile(streak_lengths, 25),
        "P50": np.percentile(streak_lengths, 50),
        "P75": np.percentile(streak_lengths, 75),
        "P90": np.percentile(streak_lengths, 90),
        "P95": np.percentile(streak_lengths, 95),
        "P99": np.percentile(streak_lengths, 99),
    }
    logger.info(f"Streak length percentiles: {percentiles}")
    return percentiles


def extract_streaks_and_multipliers(df: pd.DataFrame, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    """
    Extract streaks and their associated multipliers from the raw data.

    Args:
        df: DataFrame with game data (Game ID, Bust)
        multiplier_threshold: Threshold for considering a multiplier as a hit

    Returns:
        DataFrame with streak information
    """
    print_info("Extracting streaks and their properties")

    # Check if DataFrame has required columns
    required_cols = ["Game ID", "Bust"]
    if not all(col in df.columns for col in required_cols):
        # If we're dealing with a transformed DataFrame, it might be the result of previous processing
        if isinstance(df, pd.DataFrame) and 'streak_number' in df.columns:
            # Already a streak DataFrame, just return it
            print_info(
                f"Using existing streak DataFrame with {len(df)} streaks")
            return df
        else:
            raise ValueError(
                f"Input DataFrame must have columns {required_cols} or already be a streak DataFrame")

    # Initialize variables
    streaks = []
    current_streak = []
    current_streak_ids = []
    streak_number = 0

    # Process each game
    for i, (game_id, bust) in enumerate(zip(df["Game ID"], df["Bust"])):
        # Add current game to the streak
        current_streak.append(bust)
        current_streak_ids.append(game_id)

        # If we hit the threshold, complete the streak
        if bust >= multiplier_threshold:
            streak_number += 1
            streak_length = len(current_streak)

            # Calculate streak properties
            streak_info = {
                'streak_number': streak_number,
                'start_game_id': current_streak_ids[0],
                'end_game_id': current_streak_ids[-1],
                'streak_length': streak_length,
                'hit_multiplier': bust,
                'mean_multiplier': np.mean(current_streak),
                'std_multiplier': np.std(current_streak) if len(current_streak) > 1 else 0,
                'max_multiplier': np.max(current_streak),
                'min_multiplier': np.min(current_streak),
                'pct_gt2': np.mean([m > 2 for m in current_streak]),
                'pct_gt5': np.mean([m > 5 for m in current_streak])
            }

            # Add all multipliers in the streak
            for j, mult in enumerate(current_streak):
                streak_info[f'multiplier_{j+1}'] = mult

            streaks.append(streak_info)

            # Reset streak
            current_streak = []
            current_streak_ids = []

    # Convert to DataFrame
    streak_df = pd.DataFrame(streaks)
    print_info(f"Extracted {len(streak_df)} complete streaks")

    return streak_df


def create_streak_features(streak_df: pd.DataFrame, lookback_window: int = 5, prediction_mode: bool = False) -> pd.DataFrame:
    """
    Create features from streak data for predicting next streak length.

    Args:
        streak_df: DataFrame with streak information
        lookback_window: Number of previous streaks to consider for features
        prediction_mode: If True, don't drop rows with missing lookback data (for prediction)

    Returns:
        DataFrame with features and target for each streak
    """
    print_info(
        f"Creating streak-based features with lookback={lookback_window}")

    # Make a copy to avoid fragmentation and SettingWithCopyWarning
    streak_df = streak_df.copy()

    # Sort by streak number to ensure correct sequence
    if 'streak_number' in streak_df.columns:
        streak_df = streak_df.sort_values('streak_number')

    # --- Efficient feature creation approach ---
    # Initialize dictionaries to hold feature columns before concat
    lagged_features = {}
    rolling_features = {}

    # Create lagged features all at once
    for col in ['streak_length', 'mean_multiplier', 'max_multiplier', 'pct_gt5']:
        if col in streak_df.columns:  # Only create features for existing columns
            for i in range(1, lookback_window + 1):
                lagged_features[f'prev{i}_{col}'] = streak_df[col].shift(i)

    # Create rolling window features all at once
    for col in ['streak_length', 'mean_multiplier', 'max_multiplier', 'pct_gt5']:
        if col in streak_df.columns:  # Only create features for existing columns
            rolling_features[f'rolling_mean_{col}'] = streak_df[col].shift(
                1).rolling(lookback_window, min_periods=1).mean()
            rolling_features[f'rolling_std_{col}'] = streak_df[col].shift(
                1).rolling(lookback_window, min_periods=2).std().fillna(0)

    # Create DataFrames from the dictionaries
    lagged_df = pd.DataFrame(lagged_features, index=streak_df.index)
    rolling_df = pd.DataFrame(rolling_features, index=streak_df.index)

    # Concat all feature sets efficiently
    features_df = pd.concat([streak_df, lagged_df, rolling_df], axis=1)

    # Drop rows with NaN values (first lookback_window rows) - but not in prediction mode
    if not prediction_mode:
        initial_rows = len(features_df)
        features_df = features_df.dropna(
            subset=[f'prev{lookback_window}_streak_length'])
        dropped_rows = initial_rows - len(features_df)
        if dropped_rows > 0:
            print_info(
                f"Dropped {dropped_rows} rows due to insufficient history for full lookback window")

    # Set the target as the current streak length
    if 'streak_length' in streak_df.columns:
        features_df.loc[:,
                        'target_streak_length'] = features_df['streak_length']

    print_info(
        f"Created {features_df.shape[1]} features from streak properties")
    print_info(f"Feature matrix shape: {features_df.shape}")

    return features_df


def prepare_features(df: pd.DataFrame,
                     window: int,
                     multiplier_threshold: float = 10.0,
                     percentiles: List[float] = [0.25, 0.50, 0.75]
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare features for machine learning model training.

    This function creates streak-based features for predicting the length of the next streak.
    The clustering is done using customizable percentiles of streak lengths.

    Args:
        df: DataFrame with game data
        window: Number of previous streaks to consider (not games)
        multiplier_threshold: Threshold for considering a multiplier as a hit
        percentiles: List of percentile boundaries for clustering

    Returns:
        Tuple of (DataFrame with features, list of feature column names)
    """
    logger.info(f"Preparing streak-based features with lookback={window}")

    # Extract streaks and their properties
    streak_df = extract_streaks_and_multipliers(df, multiplier_threshold)

    # Create features based on streak patterns
    features_df = create_streak_features(streak_df, lookback_window=window)

    # Identify the feature columns (exclude target and metadata columns)
    metadata_cols = ['streak_number', 'start_game_id', 'end_game_id',
                     'streak_length', 'hit_multiplier', 'target_streak_length']
    # Also exclude the multiplier_X columns
    multiplier_cols = [
        col for col in features_df.columns if col.startswith('multiplier_')]

    feature_cols = [col for col in features_df.columns
                    if col not in metadata_cols and col not in multiplier_cols]

    # Add target clustering based on percentiles
    percentile_values = [
        features_df['target_streak_length'].quantile(p) for p in percentiles]

    # Log percentile boundaries
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val:.1f}" for p, val in zip(percentiles, percentile_values)])
    logger.info(f"Streak length percentile boundaries: {percentile_info}")

    # Create conditions and results for clustering
    conditions = []
    results = []

    for i in range(len(percentiles) + 1):
        if i == 0:
            # First cluster: <= first percentile
            conditions.append(
                features_df['target_streak_length'] <= percentile_values[0])
        elif i == len(percentiles):
            # Last cluster: > last percentile
            conditions.append(
                features_df['target_streak_length'] > percentile_values[-1])
        else:
            # Middle clusters: between adjacent percentiles
            conditions.append(
                (features_df['target_streak_length'] > percentile_values[i-1]) &
                (features_df['target_streak_length'] <= percentile_values[i])
            )
        results.append(i)

    # Create clusters (0 to n, where n is the number of percentile boundaries + 1)
    features_df = features_df.copy()  # Avoid fragmentation
    features_df.loc[:, 'target_cluster'] = np.select(
        conditions, results, default=np.nan)

    # Feature scaling - create a new DataFrame to avoid modifying the original
    scaler = StandardScaler()
    features_df.loc[:, feature_cols] = scaler.fit_transform(
        features_df[feature_cols])

    # Log cluster counts
    clusters = list(range(len(percentiles) + 1))
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


def make_feature_vector(last_streaks: List[Dict], window: int, feature_cols: List[str]) -> pd.Series:
    """
    Create a feature vector from recent streaks for prediction.

    Args:
        last_streaks: List of dictionaries containing recent streak information
        window: Number of previous streaks to consider
        feature_cols: List of feature column names to include

    Returns:
        Series with feature values
    """
    # Convert list of streak dicts to DataFrame
    streak_df = pd.DataFrame(last_streaks)

    if len(streak_df) == 0:
        # If no streaks provided, return zeros for all expected features
        logger.warning(
            "No streaks provided for feature creation. Using zeros.")
        return pd.Series(0.0, index=feature_cols)

    # Create the same features as in training, but in prediction mode to keep rows even with missing lags
    features_df = create_streak_features(
        streak_df, lookback_window=window, prediction_mode=True)

    # Get the last row which contains features for the most recent streaks
    if not features_df.empty:
        last_features = features_df.iloc[-1]

        # Create aligned feature vector with expected column names
        aligned_features = pd.Series(0.0, index=feature_cols)

        # Update values where features exist in both
        for col in feature_cols:
            if col in last_features:
                aligned_features[col] = last_features[col]

        return aligned_features
    else:
        # If we don't have enough data, create a Series with zeros
        logger.warning(
            "Failed to create feature matrix. Using zeros for all features.")
        return pd.Series(0.0, index=feature_cols)
