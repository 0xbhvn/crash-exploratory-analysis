#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data splitting functionality for temporal analysis.
"""

import pandas as pd
from typing import List, Tuple
from logger_config import (
    print_info, create_table, add_table_row, display_table
)


def temporal_train_test_split(features_df: pd.DataFrame, feature_cols: List[str], test_size: float = 0.2) -> Tuple:
    """
    Split data while maintaining strict temporal separation.

    Args:
        features_df: DataFrame with features and targets
        feature_cols: Columns to use as features
        test_size: Fraction of data to use for testing

    Returns:
        Training and testing data splits (X_train, y_train, X_test, y_test, test_indices)
    """
    print_info(
        f"Performing temporal train-test split with test_size={test_size}")

    # Sort by temporal index to ensure correct ordering
    features_df = features_df.sort_values('temporal_idx')

    # Drop rows with NaN in features or target
    valid_mask = ~features_df[feature_cols +
                              ['target_cluster']].isna().any(axis=1)
    features_df = features_df[valid_mask].copy()

    # Determine split point based on temporal_idx
    split_idx = int(len(features_df) * (1 - test_size))
    split_temporal_idx = features_df.iloc[split_idx]['temporal_idx']

    # Create train and test sets
    train_df = features_df[features_df['temporal_idx'] < split_temporal_idx]
    test_df = features_df[features_df['temporal_idx'] >= split_temporal_idx]

    # Extract features and targets
    X_train = train_df[feature_cols]
    y_train = train_df['target_cluster']
    X_test = test_df[feature_cols]
    y_test = test_df['target_cluster']

    # Save test indices for later analysis
    test_indices = test_df.index

    # Display split information
    _display_split_info(features_df, train_df, test_df,
                        test_size, split_temporal_idx)

    # Display cluster distribution
    _display_cluster_distribution(y_train, y_test)

    return X_train, y_train, X_test, y_test, test_indices


def _display_split_info(features_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame,
                        test_size: float, split_temporal_idx: int) -> None:
    """
    Display information about the train-test split.

    Args:
        features_df: Full feature DataFrame
        train_df: Training DataFrame
        test_df: Testing DataFrame
        test_size: Fraction used for testing
        split_temporal_idx: Temporal index used for splitting
    """
    # Create a table showing the split information
    split_table = create_table("Temporal Split Information",
                               ["Metric", "Value"])
    add_table_row(split_table, ["Training Samples", f"{len(train_df)}"])
    add_table_row(split_table, ["Testing Samples", f"{len(test_df)}"])
    add_table_row(split_table, ["Training %", f"{100*(1-test_size):.1f}%"])
    add_table_row(split_table, ["Testing %", f"{100*test_size:.1f}%"])
    add_table_row(split_table, [
        "Train Date Range", f"{features_df['temporal_idx'].min()} - {split_temporal_idx-1}"])
    add_table_row(split_table, [
        "Test Date Range", f"{split_temporal_idx} - {features_df['temporal_idx'].max()}"])
    display_table(split_table)


def _display_cluster_distribution(y_train: pd.Series, y_test: pd.Series) -> None:
    """
    Display distribution of clusters in train and test sets.

    Args:
        y_train: Training target values
        y_test: Testing target values
    """
    # Show distribution of clusters in train and test sets
    cluster_train = pd.Series(y_train).value_counts(
        normalize=True).sort_index() * 100
    cluster_test = pd.Series(y_test).value_counts(
        normalize=True).sort_index() * 100

    # Create a table showing cluster distribution
    cluster_table = create_table("Cluster Distribution (%)",
                                 ["Cluster", "Training Set", "Testing Set"])

    for i in range(4):  # Assuming 4 clusters
        cluster_name = f"Cluster {i}"
        train_pct = f"{cluster_train.get(i, 0):.1f}%"
        test_pct = f"{cluster_test.get(i, 0):.1f}%"
        add_table_row(cluster_table, [cluster_name, train_pct, test_pct])

    display_table(cluster_table)
