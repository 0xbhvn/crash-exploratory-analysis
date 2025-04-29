#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing module for Crash Game 10× Streak Analysis.

This module handles data loading, cleaning, and feature engineering.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Import rich logging
from utils.logger_config import (
    print_info, create_stats_table
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
