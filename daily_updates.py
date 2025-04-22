#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Daily updates module for Crash Game 10× Streak Analysis.

This module handles updating the model with new data and detecting distribution drift.
"""

import os
import logging
import pandas as pd
import datetime as dt
from typing import Dict, List, Tuple, Optional

# Import rich logging
from logger_config import (
    console, create_table, display_table, add_table_row,
    create_stats_table, print_info, print_success, print_warning,
    print_error, print_panel
)

logger = logging.getLogger(__name__)


def process_daily_update(analyzer, new_rows: pd.DataFrame, drift_threshold: float = 0.005) -> bool:
    """
    Process new game data, update model if drift is detected.

    Args:
        analyzer: CrashStreakAnalyzer instance
        new_rows: DataFrame with new game data
        drift_threshold: Threshold for detecting drift in 10× rate

    Returns:
        Boolean indicating whether model was retrained
    """
    print_info(f"Processing {len(new_rows)} new rows for daily update")

    # Update dataframe
    analyzer.df = pd.concat([analyzer.df, new_rows]).reset_index(drop=True)

    # Recompute empirical 10× rate
    p_today = (new_rows['Bust'] >= 10).mean()

    # Display drift statistics
    drift_stats = {
        "New Rows": len(new_rows),
        "First Game ID": new_rows["Game ID"].min() if not new_rows.empty else "N/A",
        "Last Game ID": new_rows["Game ID"].max() if not new_rows.empty else "N/A",
        "Current 10× Rate": f"{p_today:.4f}",
        "Historical 10× Rate": f"{analyzer.p_hat:.4f}",
        "Absolute Difference": f"{abs(p_today - analyzer.p_hat):.4f}",
        "Drift Threshold": f"{drift_threshold:.4f}"
    }
    create_stats_table("Drift Analysis", drift_stats)

    if abs(p_today - analyzer.p_hat) > drift_threshold:
        print_warning(
            f"Drift detected: today's 10× rate {p_today:.4f} vs. historical {analyzer.p_hat:.4f}")
        print_info("Retraining model...")

        # Re-prepare features and retrain
        analyzer.prepare_features()
        analyzer.train_model()

        # Update p_hat
        analyzer.p_hat = (analyzer.df['Bust'] >= 10).mean()
        print_success(f"Updated model with new 10× rate: {analyzer.p_hat:.4f}")
        return True
    else:
        print_info(
            f"No significant drift detected ({p_today:.4f} vs {analyzer.p_hat:.4f})")
        return False


def save_model_snapshot(analyzer, output_dir: str = None) -> None:
    """
    Save a snapshot of the current model and data.

    Args:
        analyzer: CrashStreakAnalyzer instance
        output_dir: Directory to save snapshot (defaults to analyzer.output_dir)
    """
    if output_dir is None:
        output_dir = analyzer.output_dir

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(output_dir, f"snapshot_{timestamp}")

    os.makedirs(snapshot_dir, exist_ok=True)

    # Save model
    import joblib
    model_path = os.path.join(snapshot_dir, "model.pkl")
    joblib.dump(analyzer.bst_final, model_path)
    print_info(f"Saved model to {model_path}")

    # Save data
    data_path = os.path.join(snapshot_dir, "data.csv")
    analyzer.df.to_csv(data_path, index=False)
    print_info(f"Saved dataset with {len(analyzer.df):,} rows to {data_path}")

    # Save metadata
    import json
    metadata = {
        "timestamp": timestamp,
        "p_hat": analyzer.p_hat,
        "window": analyzer.WINDOW,
        "clusters": {str(k): v for k, v in analyzer.CLUSTERS.items()},
        "test_frac": analyzer.TEST_FRAC,
        "data_rows": len(analyzer.df)
    }

    meta_path = os.path.join(snapshot_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print_info(f"Saved metadata to {meta_path}")

    # Display snapshot summary
    snapshot_info = {
        "Timestamp": timestamp,
        "Model Path": model_path,
        "Data Path": data_path,
        "Metadata Path": meta_path,
        "Total Rows": len(analyzer.df),
        "10× Rate": f"{analyzer.p_hat:.4f}",
        "Snapshot Directory": snapshot_dir
    }
    create_stats_table("Model Snapshot Summary", snapshot_info)

    print_success(f"Saved model snapshot to {snapshot_dir}")


def load_new_data(data_path: str) -> pd.DataFrame:
    """
    Load new data for daily update.

    Args:
        data_path: Path to CSV file with new game data

    Returns:
        DataFrame with new game data
    """
    print_info(f"Loading new data from {data_path}")

    try:
        df_new = pd.read_csv(data_path)

        # Basic validation
        required_columns = ["Game ID", "Bust"]
        for col in required_columns:
            if col not in df_new.columns:
                print_error(f"Required column '{col}' not found in new data")
                raise ValueError(
                    f"Required column '{col}' not found in new data")

        # Clean data
        df_new = (
            df_new
            .dropna()
            .rename(columns=lambda c: c.strip())
        )
        df_new["Game ID"] = df_new["Game ID"].astype(int)
        df_new["Bust"] = df_new["Bust"].astype(float)

        # Display summary of new data
        new_data_stats = {
            "Total New Rows": len(df_new),
            "First Game ID": df_new["Game ID"].min() if not df_new.empty else "N/A",
            "Last Game ID": df_new["Game ID"].max() if not df_new.empty else "N/A",
            "Min Multiplier": df_new["Bust"].min() if not df_new.empty else "N/A",
            "Max Multiplier": df_new["Bust"].max() if not df_new.empty else "N/A",
            "Avg Multiplier": round(df_new["Bust"].mean(), 2) if not df_new.empty else "N/A",
            "10× or Higher": (df_new["Bust"] >= 10).sum() if not df_new.empty else "N/A",
            "10× Rate": f"{(df_new['Bust'] >= 10).mean() * 100:.2f}%" if not df_new.empty else "N/A"
        }
        create_stats_table("New Data Summary", new_data_stats)

        # Display a sample of the new data
        if not df_new.empty:
            sample_table = create_table(
                "Sample of New Data", ["Game ID", "Bust"])
            for _, row in df_new.head(5).iterrows():
                add_table_row(
                    sample_table, [row["Game ID"], f"{row['Bust']:.2f}"])
            display_table(sample_table)

        print_success(f"Successfully loaded {len(df_new)} new rows")
        return df_new

    except Exception as e:
        print_error(f"Error loading new data: {str(e)}")
        raise
