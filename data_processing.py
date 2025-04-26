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
                     ) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    """
    Prepare features for machine learning using streak-based analysis.

    WARNING: This function can create data leakage in a prediction context 
    because it processes the entire dataset at once. Use prepare_train_test_features instead.

    Args:
        df: DataFrame with game data (Game ID, Bust)
        window: Number of previous streaks to consider for features
        multiplier_threshold: Threshold for considering a multiplier as a hit
        percentiles: List of percentile boundaries for clustering

    Returns:
        Tuple of (DataFrame with features and target, list of feature column names, fitted StandardScaler)
    """
    logger.warning(
        "Using prepare_features which may create data leakage. Consider using prepare_train_test_features instead.")
    print_info(f"Preparing streak-based features with lookback={window}")

    # Extract streaks
    streak_df = extract_streaks_and_multipliers(df, multiplier_threshold)

    # Create features from streaks
    features_df = create_streak_features(streak_df, lookback_window=window)

    # Calculate streak length percentiles
    features_df['target_streak_length'] = features_df['streak_length']
    percentile_values = [
        features_df['target_streak_length'].quantile(p) for p in percentiles]

    # Log percentile boundaries
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val}" for p, val in zip(percentiles, percentile_values)])
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

    # Create clusters
    features_df = features_df.copy()  # Avoid fragmentation
    features_df['target_cluster'] = np.select(
        conditions, results, default=np.nan)

    # Log cluster distribution
    cluster_counts = []
    for i in range(len(percentiles) + 1):
        count = (features_df['target_cluster'] == i).sum()
        percentage = count / len(features_df) * 100
        cluster_counts.append(f"{i}: {count} ({percentage:.1f}%)")

    logger.info(
        f"Percentile-based cluster counts: {', '.join(cluster_counts)}")

    # Identify the feature columns (exclude target and metadata columns)
    metadata_cols = ['streak_number', 'start_game_id', 'end_game_id',
                     'streak_length', 'hit_multiplier', 'target_streak_length']
    # Also exclude the multiplier_X columns
    multiplier_cols = [
        col for col in features_df.columns if col.startswith('multiplier_')]

    feature_cols = [col for col in features_df.columns
                    if col not in metadata_cols and col not in multiplier_cols
                    and col != 'target_cluster']  # Explicitly exclude target_cluster

    # Apply scaling to features
    scaler = StandardScaler()
    features_df[feature_cols] = scaler.fit_transform(features_df[feature_cols])

    logger.info(f"Final feature matrix shape: {features_df.shape}")
    logger.info(
        f"StandardScaler mean and scale computed for {len(feature_cols)} features")

    return features_df, feature_cols, scaler


def prepare_train_test_features(df: pd.DataFrame,
                                window: int,
                                test_frac: float,
                                random_seed: int = 42,
                                multiplier_threshold: float = 10.0,
                                percentiles: List[float] = [0.25, 0.50, 0.75]
                                ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], StandardScaler]:
    """
    Prepare features for machine learning with proper train-test split to prevent data leakage.

    This function first extracts streaks from raw game data, then splits the streaks into
    training and testing sets, and finally applies feature engineering to each set separately.

    Args:
        df: DataFrame with game data (Game ID, Bust)
        window: Number of previous streaks to consider for lookback features
        test_frac: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        multiplier_threshold: Threshold for considering a multiplier as a hit
        percentiles: List of percentile boundaries for clustering

    Returns:
        Tuple of (train DataFrame with features, test DataFrame with features, 
                 list of feature column names, fitted StandardScaler)
    """
    logger.info(
        f"Preparing train-test features with lookback={window} and test_frac={test_frac}")

    # Step 1: Extract all streaks from raw game data
    streak_df = extract_streaks_and_multipliers(df, multiplier_threshold)

    # Step 2: Split streaks into train and test sets BEFORE creating features
    # Sort by streak_number to ensure temporal order
    streak_df = streak_df.sort_values('streak_number')

    # Use sequential split by index to ensure we don't break streak continuity
    split_idx = int(len(streak_df) * (1 - test_frac))
    train_streak_df = streak_df.iloc[:split_idx].copy()
    test_streak_df = streak_df.iloc[split_idx:].copy()

    # Show split information
    split_info = {
        "Training Streaks": len(train_streak_df),
        "Testing Streaks": len(test_streak_df),
        "Training Split": f"{(1-test_frac)*100:.1f}%",
        "Testing Split": f"{test_frac*100:.1f}%"
    }
    create_stats_table("Streak Train/Test Split", split_info)

    # Step 3: Create features for training data
    logger.info("Creating features for training data...")
    train_features_df = create_streak_features(
        train_streak_df, lookback_window=window)

    # Identify the feature columns (exclude target and metadata columns)
    metadata_cols = ['streak_number', 'start_game_id', 'end_game_id',
                     'streak_length', 'hit_multiplier', 'target_streak_length']
    # Also exclude the multiplier_X columns
    multiplier_cols = [
        col for col in train_features_df.columns if col.startswith('multiplier_')]

    feature_cols = [col for col in train_features_df.columns
                    if col not in metadata_cols and col not in multiplier_cols]

    # Step 4: Calculate streak length percentiles from training data only
    train_features_df.loc[:,
                          'target_streak_length'] = train_features_df['streak_length']
    percentile_values = [
        train_features_df['target_streak_length'].quantile(p) for p in percentiles]

    # Log percentile boundaries
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val:.1f}" for p, val in zip(percentiles, percentile_values)])
    logger.info(
        f"Streak length percentile boundaries from training data: {percentile_info}")

    # Step 5: Apply clustering to training data based on percentiles
    # Create conditions and results for clustering
    conditions = []
    results = []

    for i in range(len(percentiles) + 1):
        if i == 0:
            # First cluster: <= first percentile
            conditions.append(
                train_features_df['target_streak_length'] <= percentile_values[0])
        elif i == len(percentiles):
            # Last cluster: > last percentile
            conditions.append(
                train_features_df['target_streak_length'] > percentile_values[-1])
        else:
            # Middle clusters: between adjacent percentiles
            conditions.append(
                (train_features_df['target_streak_length'] > percentile_values[i-1]) &
                (train_features_df['target_streak_length']
                 <= percentile_values[i])
            )
        results.append(i)

    # Create clusters
    train_features_df = train_features_df.copy()  # Avoid fragmentation
    train_features_df.loc[:, 'target_cluster'] = np.select(
        conditions, results, default=np.nan)

    # Step 6: Feature scaling - fit on training data only
    scaler = StandardScaler()
    train_features_df.loc[:, feature_cols] = scaler.fit_transform(
        train_features_df[feature_cols])

    # Step 7: Create features for test data separately
    # IMPORTANT: We need to handle the case where features depend on data from training set
    # Add buffer rows from training set to ensure features for first test rows are complete
    logger.info("Creating features for test data...")

    if window > 0:
        # Take the last 'window' rows from training set as buffer
        buffer_rows = min(window, len(train_streak_df))
        buffer_streak_df = train_streak_df.iloc[-buffer_rows:].copy()

        # Combine buffer with test data for feature creation
        combined_streak_df = pd.concat([buffer_streak_df, test_streak_df])

        # Create features using the combined data
        combined_features_df = create_streak_features(
            combined_streak_df, lookback_window=window)

        # Only keep the test portion for the final test dataframe
        test_features_df = combined_features_df.iloc[buffer_rows:].copy()
    else:
        # If no lookback window, just process test data directly
        test_features_df = create_streak_features(
            test_streak_df, lookback_window=window)

    # Add target streak length to test data
    test_features_df.loc[:,
                         'target_streak_length'] = test_features_df['streak_length']

    # Step 8: Apply the same clustering logic to test data using TRAINING percentiles
    conditions = []
    results = []

    for i in range(len(percentiles) + 1):
        if i == 0:
            conditions.append(
                test_features_df['target_streak_length'] <= percentile_values[0])
        elif i == len(percentiles):
            conditions.append(
                test_features_df['target_streak_length'] > percentile_values[-1])
        else:
            conditions.append(
                (test_features_df['target_streak_length'] > percentile_values[i-1]) &
                (test_features_df['target_streak_length']
                 <= percentile_values[i])
            )
        results.append(i)

    test_features_df = test_features_df.copy()
    test_features_df.loc[:, 'target_cluster'] = np.select(
        conditions, results, default=np.nan)

    # Step 9: Apply the same scaling to test data using the scaler fit on training data
    for col in feature_cols:
        if col not in test_features_df.columns:
            logger.warning(
                f"Missing feature column in test set: {col}. Adding with zeros.")
            test_features_df[col] = 0

    # Fill any NaN values with zeros
    test_features_df[feature_cols] = test_features_df[feature_cols].fillna(0)

    # Transform test features using the scaler fitted on training data
    test_features_df.loc[:, feature_cols] = scaler.transform(
        test_features_df[feature_cols])

    # Log cluster counts for both sets
    train_cluster_counts = []
    for i in range(len(percentiles) + 1):
        count = (train_features_df['target_cluster'] == i).sum()
        percentage = count / len(train_features_df) * 100
        train_cluster_counts.append(f"{i}: {count} ({percentage:.1f}%)")

    test_cluster_counts = []
    for i in range(len(percentiles) + 1):
        count = (test_features_df['target_cluster'] == i).sum()
        percentage = count / len(test_features_df) * 100
        test_cluster_counts.append(f"{i}: {count} ({percentage:.1f}%)")

    logger.info(f"Training cluster counts: {', '.join(train_cluster_counts)}")
    logger.info(f"Testing cluster counts: {', '.join(test_cluster_counts)}")

    return train_features_df, test_features_df, feature_cols, scaler


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
    # Validate inputs
    if feature_cols is None or len(feature_cols) == 0:
        logger.error("No feature columns provided to make_feature_vector")
        return None

    logger.info(
        f"Creating feature vector with {len(last_streaks)} streaks and {len(feature_cols)} feature columns")

    # Convert list of streak dicts to DataFrame
    streak_df = pd.DataFrame(last_streaks)

    if len(streak_df) == 0:
        # If no streaks provided, return zeros for all expected features
        logger.warning(
            "No streaks provided for feature creation. Using zeros.")
        return pd.Series(0.0, index=feature_cols)

    # Log streaks info
    if 'streak_length' in streak_df.columns:
        logger.info(f"Streak lengths: min={streak_df['streak_length'].min()}, "
                    f"max={streak_df['streak_length'].max()}, "
                    f"mean={streak_df['streak_length'].mean():.2f}")

    # Check for required columns
    required_cols = ['streak_length', 'mean_multiplier',
                     'max_multiplier', 'min_multiplier']
    missing_cols = [
        col for col in required_cols if col not in streak_df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in streak data: {missing_cols}")

    # Create the same features as in training, but in prediction mode to keep rows even with missing lags
    features_df = create_streak_features(
        streak_df, lookback_window=window, prediction_mode=True)

    logger.info(f"Feature DataFrame shape: {features_df.shape}")

    # Get the last row which contains features for the most recent streaks
    if not features_df.empty:
        last_features = features_df.iloc[-1]
        logger.info(f"Created {len(last_features)} features for last streak")

        # Create aligned feature vector with expected column names
        aligned_features = pd.Series(0.0, index=feature_cols)

        # Update values where features exist in both
        matched_features = []
        for col in feature_cols:
            if col in last_features:
                try:
                    # Extract scalar value if needed
                    val = last_features[col]
                    # Check if value is array-like and extract first element if needed
                    if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                        logger.warning(
                            f"Feature '{col}' has sequence value, extracting first element")
                        if len(val) > 0:
                            aligned_features[col] = float(val[0])
                        else:
                            aligned_features[col] = 0.0
                    else:
                        # Handle scalar values
                        aligned_features[col] = float(val)
                    matched_features.append(col)
                except Exception as e:
                    logger.warning(
                        f"Could not convert feature '{col}' to float: {e}, using 0.0")
                    aligned_features[col] = 0.0

        # Log alignment statistics
        match_rate = len(matched_features) / len(feature_cols) * 100
        logger.info(
            f"Feature alignment: {len(matched_features)}/{len(feature_cols)} features matched ({match_rate:.1f}%)")

        # Log some key features if available
        key_features = ['mean_multiplier', 'max_multiplier', 'min_multiplier', 'pct_gt5', 'pct_gt2',
                        'rolling_mean_streak_length', 'rolling_std_streak_length']
        matched_key_features = [
            f for f in key_features if f in matched_features]
        if matched_key_features:
            logger.info(
                f"Key matched features: {', '.join(f'{f}={aligned_features[f]:.4f}' for f in matched_key_features)}")

        return aligned_features
    else:
        # If we don't have enough data, create a Series with zeros
        logger.warning(
            "Failed to create feature matrix. Using zeros for all features.")
        return pd.Series(0.0, index=feature_cols)
