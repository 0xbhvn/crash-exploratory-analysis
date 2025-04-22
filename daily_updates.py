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
    logger.info(f"Processing {len(new_rows)} new rows for daily update")

    # Update dataframe
    analyzer.df = pd.concat([analyzer.df, new_rows]).reset_index(drop=True)

    # Recompute empirical 10× rate
    p_today = (new_rows['Bust'] >= 10).mean()

    if abs(p_today - analyzer.p_hat) > drift_threshold:
        logger.warning(
            f"Drift detected: today's 10× rate {p_today:.4f} vs. historical {analyzer.p_hat:.4f}")
        logger.info("Retraining model...")

        # Re-prepare features and retrain
        analyzer.prepare_features()
        analyzer.train_model()

        # Update p_hat
        analyzer.p_hat = (analyzer.df['Bust'] >= 10).mean()
        logger.info(f"Updated model with new 10× rate: {analyzer.p_hat:.4f}")
        return True
    else:
        logger.info(
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
    joblib.dump(analyzer.bst_final, os.path.join(snapshot_dir, "model.pkl"))

    # Save data
    analyzer.df.to_csv(os.path.join(snapshot_dir, "data.csv"), index=False)

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

    with open(os.path.join(snapshot_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Saved model snapshot to {snapshot_dir}")


def load_new_data(data_path: str) -> pd.DataFrame:
    """
    Load new data for daily update.

    Args:
        data_path: Path to CSV file with new game data

    Returns:
        DataFrame with new game data
    """
    logger.info(f"Loading new data from {data_path}")

    try:
        df_new = pd.read_csv(data_path)

        # Basic validation
        required_columns = ["Game ID", "Bust"]
        for col in required_columns:
            if col not in df_new.columns:
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

        logger.info(f"Loaded {len(df_new)} new rows")
        return df_new

    except Exception as e:
        logger.error(f"Error loading new data: {str(e)}")
        raise
