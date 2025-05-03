#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature engineering functionality for temporal analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from utils.logger_config import (
    print_info, create_table, add_table_row, display_table
)


def create_temporal_features(streak_df: pd.DataFrame, lookback_window: int = 5, verbose: bool = True) -> Tuple[pd.DataFrame, List[str], List[float]]:
    """
    Create strictly temporal features with no leakage.

    Args:
        streak_df: DataFrame with streak information
        lookback_window: Number of previous streaks to use for features
        verbose: If True, print detailed logging messages.

    Returns:
        DataFrame with temporal features, feature column names, and percentile values
    """
    if verbose:
        print_info(
            f"Creating strictly temporal features with lookback={lookback_window}")

    # Make a copy to avoid SettingWithCopyWarning
    features_df = streak_df.copy()

    # Create all temporal features
    features_df = _create_lagged_features(features_df, lookback_window)
    features_df = _create_rolling_features(features_df, lookback_window)
    features_df = _create_category_features(features_df)
    features_df = _create_time_since_features(features_df)
    features_df = _create_timestamp_features(features_df, streak_df)
    features_df = _create_one_hot_features(features_df)

    # Get valid feature columns that don't cause data leakage
    feature_cols = _get_valid_feature_columns(features_df, verbose=verbose)

    if verbose:
        print_info(f"Created {len(feature_cols)} strictly temporal features")

    # Calculate percentile values for target creation
    percentile_values = _calculate_percentiles(streak_df, verbose=verbose)

    # Create target clusters
    features_df = _create_target_clusters(
        features_df, streak_df, percentile_values)

    # Display example features table
    if verbose:
        _display_feature_examples()

    return features_df, feature_cols, percentile_values


def _create_lagged_features(features_df: pd.DataFrame, lookback_window: int) -> pd.DataFrame:
    """
    Create lagged features based on previous streak data.

    Args:
        features_df: DataFrame to add features to
        lookback_window: Number of previous streaks to use

    Returns:
        DataFrame with lagged features added
    """
    # Lagged features - focus on streak lengths and patterns, not current streak properties
    for i in range(1, lookback_window + 1):
        features_df[f'prev{i}_length'] = features_df['streak_length'].shift(i)
        features_df[f'prev{i}_hit_mult'] = features_df['hit_multiplier'].shift(
            i)

        # Create streak length difference features
        if i > 1:
            features_df[f'diff{i-1}_to_{i}'] = features_df[f'prev{i-1}_length'] - \
                features_df[f'prev{i}_length']

    return features_df


def _create_rolling_features(features_df: pd.DataFrame, lookback_window: int) -> pd.DataFrame:
    """
    Create rolling window statistical features.

    Args:
        features_df: DataFrame to add features to
        lookback_window: Maximum lookback window

    Returns:
        DataFrame with rolling features added
    """
    for window in [3, 5, 10]:
        if window <= lookback_window:
            # Use shifted data to ensure we only look at past values
            features_df[f'rolling_mean_{window}'] = features_df['streak_length'].shift(
                1).rolling(window, min_periods=1).mean()
            features_df[f'rolling_std_{window}'] = features_df['streak_length'].shift(
                1).rolling(window, min_periods=2).std().fillna(0)
            features_df[f'rolling_max_{window}'] = features_df['streak_length'].shift(
                1).rolling(window, min_periods=1).max()
            features_df[f'rolling_min_{window}'] = features_df['streak_length'].shift(
                1).rolling(window, min_periods=1).min()

            # Track streaks by category
            features_df[f'short_pct_{window}'] = (
                (features_df['streak_length'].shift(1) <= 3).rolling(window, min_periods=1).mean())
            features_df[f'medium_pct_{window}'] = (((features_df['streak_length'].shift(1) > 3) &
                                                   (features_df['streak_length'].shift(1) <= 14))
                                                   .rolling(window, min_periods=1).mean())
            features_df[f'long_pct_{window}'] = ((features_df['streak_length'].shift(1) > 14)
                                                 .rolling(window, min_periods=1).mean())

            # New feature: rolling trend of hit multipliers
            features_df[f'hit_mult_mean_{window}'] = features_df['hit_multiplier'].shift(
                1).rolling(window, min_periods=1).mean()
            features_df[f'hit_mult_trend_{window}'] = features_df[f'hit_mult_mean_{window}'] - \
                features_df['hit_multiplier'].shift(window+1)

    return features_df


def _create_category_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create streak category features and pattern recognition features.

    Args:
        features_df: DataFrame to add features to

    Returns:
        DataFrame with category features added
    """
    # Calculate streak patterns (transitions between streak lengths)
    features_df['streak_category'] = pd.cut(
        features_df['streak_length'],
        bins=[0, 3, 7, 14, float('inf')],
        labels=['short', 'medium_short', 'medium_long', 'long'],
        right=True
    )

    # Convert to categorical for easier manipulation
    features_df['streak_category'] = features_df['streak_category'].astype(
        'category')

    # Track patterns of similar categories
    features_df['prev_category'] = features_df['streak_category'].shift(1)
    features_df['same_as_prev'] = (
        features_df['streak_category'] == features_df['prev_category']).astype(int)

    # Count consecutive similar categories
    run_counter = 0
    run_counts = []
    prev_cat = None

    for cat in features_df['streak_category']:
        if cat == prev_cat:
            run_counter += 1
        else:
            run_counter = 1
            prev_cat = cat
        run_counts.append(run_counter)

    features_df['category_run_length'] = run_counts
    features_df['prev_run_length'] = features_df['category_run_length'].shift(
        1).fillna(0).astype(int)

    return features_df


def _create_time_since_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-since-last-category features.

    Args:
        features_df: DataFrame to add features to

    Returns:
        DataFrame with time-since features added
    """
    # New feature: time-since-last-category
    for category in ['short', 'medium_short', 'medium_long', 'long']:
        # Initialize counters
        counter = []
        last_seen = -1

        # Iterate through all rows in forward order
        for i, prev_cat in enumerate(features_df['prev_category']):
            if pd.isna(prev_cat):
                # If prev_category is NaN (first row), set counter to a high value
                counter.append(99)
            elif prev_cat == category:
                # Reset counter if category matches
                last_seen = i
                counter.append(0)
            else:
                # Increment counter by 1 if seen before, else high value
                counter.append(i - last_seen if last_seen >= 0 else 99)

        # Add the counter as a feature
        features_df[f'time_since_{category}'] = counter

    return features_df


def _create_timestamp_features(features_df: pd.DataFrame, streak_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create timestamp-based features if available.

    Args:
        features_df: DataFrame to add features to
        streak_df: Original streak DataFrame that may contain timestamp

    Returns:
        DataFrame with timestamp features added if available
    """
    # Add day of week and hour features (from Game ID, if available)
    if 'timestamp' in streak_df.columns:
        features_df['hour'] = streak_df['timestamp'].dt.hour
        features_df['day_of_week'] = streak_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = (
            features_df['day_of_week'] >= 5).astype(int)

    return features_df


def _create_one_hot_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one-hot encoded features for categorical variables.

    Args:
        features_df: DataFrame to add features to

    Returns:
        DataFrame with one-hot encoded features added
    """
    # Create one-hot encoded features for categorical variables
    features_df = pd.get_dummies(
        features_df,
        columns=['prev_category'],
        prefix=['prev_cat'],
        drop_first=False
    )

    return features_df


def _get_valid_feature_columns(features_df: pd.DataFrame, verbose: bool = True) -> List[str]:
    """
    Get valid feature columns that won't cause data leakage.

    Args:
        features_df: DataFrame with features
        verbose: If True, print logging messages.

    Returns:
        List of valid feature column names
    """
    # Create a clean list of feature columns, excluding current streak properties
    exclude_cols = [
        'streak_number', 'start_game_id', 'end_game_id', 'streak_length',
        'hit_multiplier', 'mean_multiplier', 'std_multiplier', 'max_multiplier', 'min_multiplier',
        'pct_gt2', 'pct_gt5', 'temporal_idx', 'streak_category'
    ]

    # Also exclude any multiplier_X columns from the current streak
    multiplier_cols = [
        col for col in features_df.columns if col.startswith('multiplier_')
    ]

    # Combine all columns to exclude
    all_exclude_cols = exclude_cols + multiplier_cols

    # Get valid feature columns
    all_cols = features_df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in all_exclude_cols]

    # Only print excluded features if verbose
    if verbose and len(exclude_cols) > 0:
        print_info(
            f"Excluded {len(exclude_cols)} non-temporal features to prevent leakage")

    return feature_cols


def _calculate_percentiles(streak_df: pd.DataFrame, verbose: bool = True) -> List[float]:
    """
    Calculate percentiles for streak length clustering.

    Args:
        streak_df: DataFrame with streak lengths
        verbose: If True, print logging messages.

    Returns:
        List of percentile values [P25, P50, P75]
    """
    # Create target columns based on percentiles
    percentiles = [0.25, 0.50, 0.75]
    percentile_values = [
        streak_df['streak_length'].quantile(p) for p in percentiles
    ]

    # Log percentile values
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val:.1f}" for p,
         val in zip(percentiles, percentile_values)]
    )
    if verbose:
        print_info(f"Streak length percentiles: {percentile_info}")

    return percentile_values


def _create_target_clusters(features_df: pd.DataFrame, streak_df: pd.DataFrame, percentile_values: List[float]) -> pd.DataFrame:
    """
    Create target clusters based on streak length percentiles.

    Args:
        features_df: DataFrame to add target clusters to
        streak_df: Original streak DataFrame with streak_length
        percentile_values: List of percentile values for clustering

    Returns:
        DataFrame with target clusters added
    """
    # Create cluster target based on percentiles
    conditions = []
    results = []

    for i in range(len(percentile_values) + 1):
        if i == 0:
            # First cluster: <= first percentile
            conditions.append(
                streak_df['streak_length'] <= percentile_values[0])
        elif i == len(percentile_values):
            # Last cluster: > last percentile
            conditions.append(
                streak_df['streak_length'] > percentile_values[-1])
        else:
            # Middle clusters: between adjacent percentiles
            conditions.append(
                (streak_df['streak_length'] > percentile_values[i-1]) &
                (streak_df['streak_length'] <= percentile_values[i])
            )
        results.append(i)

    # Create target cluster
    features_df['target_cluster'] = np.select(
        conditions, results, default=np.nan)

    return features_df


def _display_feature_examples() -> None:
    """
    Display a table with examples of temporal features.
    """
    # Create a table showing the temporal features
    feature_table = create_table("Temporal Feature Examples (First 10)",
                                 ["Feature", "Description", "Type"])

    # Add some example features to the table
    add_table_row(feature_table, [
        "prev1_length", "Streak length of the previous streak", "Lag"])
    add_table_row(feature_table, [
        "prev2_length", "Streak length from 2 streaks ago", "Lag"])
    add_table_row(feature_table, [
        "diff1_to_2", "Difference between prev1 and prev2 lengths", "Difference"])
    add_table_row(feature_table, [
        "rolling_mean_5", "Average of the last 5 streak lengths", "Rolling"])
    add_table_row(feature_table, [
        "short_pct_5", "Percentage of short streaks in last 5", "Pattern"])
    add_table_row(feature_table, [
        "hit_mult_mean_5", "Average of the last 5 hit multipliers", "Rolling"])
    add_table_row(feature_table, [
        "hit_mult_trend_5", "Trend in hit multipliers over last 5", "Trend"])
    add_table_row(feature_table, [
        "time_since_long", "Streaks since last long streak", "Pattern"])
    add_table_row(feature_table, [
        "category_run_length", "Count of consecutive streaks in same category", "Pattern"])
    add_table_row(feature_table, [
        "same_as_prev", "Whether current category same as previous", "Pattern"])

    display_table(feature_table)
