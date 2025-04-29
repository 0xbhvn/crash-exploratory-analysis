#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data loading functionality for temporal analysis.
"""

import pandas as pd
from logger_config import (
    print_info, create_stats_table
)
from data_processing import extract_streaks_and_multipliers


def load_data(csv_path: str, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    """
    Load and extract streaks from crash game data.

    Args:
        csv_path: Path to CSV file with game data
        multiplier_threshold: Threshold for streak hits

    Returns:
        DataFrame with streak information
    """
    print_info(f"Loading data from {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    df["Game ID"] = df["Game ID"].astype(int)
    df["Bust"] = df["Bust"].astype(float)

    print_info(f"Loaded {len(df)} games, extracting streaks")

    # Extract streaks
    streak_df = extract_streaks_and_multipliers(df, multiplier_threshold)

    print_info(f"Extracted {len(streak_df)} complete streaks")

    # Add a sequential index that respects the temporal ordering
    streak_df['temporal_idx'] = range(len(streak_df))

    # Display a summary of the data
    _display_data_summary(streak_df)

    return streak_df


def _display_data_summary(streak_df: pd.DataFrame) -> None:
    """
    Display a summary of the streak data.

    Args:
        streak_df: DataFrame with streak information
    """
    data_stats = {
        "Total Streaks": len(streak_df),
        "Min Streak Length": streak_df['streak_length'].min(),
        "Max Streak Length": streak_df['streak_length'].max(),
        "Mean Streak Length": f"{streak_df['streak_length'].mean():.2f}",
        "First Game ID": streak_df['start_game_id'].min(),
        "Last Game ID": streak_df['end_game_id'].max()
    }

    create_stats_table("Data Summary", data_stats)
