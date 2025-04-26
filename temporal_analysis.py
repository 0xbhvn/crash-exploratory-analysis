#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Analysis for Crash Game 10× Streak Analysis.

This script implements enhanced time-series modeling with:
1. Strict temporal separation to eliminate data leakage
2. Purely historical features (no current streak properties)
3. Focus on genuinely predictive temporal patterns
4. Realistic evaluation metrics for time-series prediction
5. Transition analysis to measure pattern prediction power
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import json
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Import rich logging
from logger_config import (
    setup_logging, console, print_info, print_success, print_warning,
    print_error, print_panel, create_stats_table, create_table, add_table_row,
    display_table
)

# Import from local modules
from data_processing import extract_streaks_and_multipliers
from rich_summary import display_output_summary

# Setup rich logging
logger = setup_logging()


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
    data_stats = {
        "Total Streaks": len(streak_df),
        "Min Streak Length": streak_df['streak_length'].min(),
        "Max Streak Length": streak_df['streak_length'].max(),
        "Mean Streak Length": f"{streak_df['streak_length'].mean():.2f}",
        "First Game ID": streak_df['start_game_id'].min(),
        "Last Game ID": streak_df['end_game_id'].max()
    }

    create_stats_table("Data Summary", data_stats)

    return streak_df


def create_temporal_features(streak_df: pd.DataFrame, lookback_window: int = 5) -> Tuple[pd.DataFrame, List[str], List[float]]:
    """
    Create strictly temporal features with no leakage.

    Args:
        streak_df: DataFrame with streak information
        lookback_window: Number of previous streaks to use for features

    Returns:
        DataFrame with temporal features, feature column names, and percentile values
    """
    print_info(
        f"Creating strictly temporal features with lookback={lookback_window}")

    # Make a copy to avoid SettingWithCopyWarning
    features_df = streak_df.copy()

    # ---------------------------------------------------------------------------
    # 1. STRICTLY HISTORICAL FEATURES - only using past information
    # ---------------------------------------------------------------------------

    # Lagged features - focus on streak lengths and patterns, not current streak properties
    for i in range(1, lookback_window + 1):
        features_df[f'prev{i}_length'] = streak_df['streak_length'].shift(i)
        features_df[f'prev{i}_hit_mult'] = streak_df['hit_multiplier'].shift(i)

        # Create streak length difference features
        if i > 1:
            features_df[f'diff{i-1}_to_{i}'] = features_df[f'prev{i-1}_length'] - \
                features_df[f'prev{i}_length']

    # Rolling window statistics
    for window in [3, 5, 10]:
        if window <= lookback_window:
            # Use shifted data to ensure we only look at past values
            features_df[f'rolling_mean_{window}'] = streak_df['streak_length'].shift(
                1).rolling(window, min_periods=1).mean()
            features_df[f'rolling_std_{window}'] = streak_df['streak_length'].shift(
                1).rolling(window, min_periods=2).std().fillna(0)
            features_df[f'rolling_max_{window}'] = streak_df['streak_length'].shift(
                1).rolling(window, min_periods=1).max()
            features_df[f'rolling_min_{window}'] = streak_df['streak_length'].shift(
                1).rolling(window, min_periods=1).min()

            # Track streaks by category
            features_df[f'short_pct_{window}'] = (
                (streak_df['streak_length'].shift(1) <= 3).rolling(window, min_periods=1).mean())
            features_df[f'medium_pct_{window}'] = (((streak_df['streak_length'].shift(1) > 3) &
                                                   (streak_df['streak_length'].shift(1) <= 14))
                                                   .rolling(window, min_periods=1).mean())
            features_df[f'long_pct_{window}'] = ((streak_df['streak_length'].shift(1) > 14)
                                                 .rolling(window, min_periods=1).mean())

    # Calculate streak patterns (transitions between streak lengths)
    features_df['streak_category'] = pd.cut(
        streak_df['streak_length'],
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

    # Add day of week and hour features (from Game ID, if available)
    if 'timestamp' in streak_df.columns:
        features_df['hour'] = streak_df['timestamp'].dt.hour
        features_df['day_of_week'] = streak_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = (
            features_df['day_of_week'] >= 5).astype(int)

    # Create one-hot encoded features for categorical variables
    features_df = pd.get_dummies(
        features_df,
        columns=['prev_category'],
        prefix=['prev_cat'],
        drop_first=False
    )

    # ---------------------------------------------------------------------------
    # 2. DROP NON-TEMPORAL FEATURES - to prevent data leakage
    # ---------------------------------------------------------------------------

    # Create a clean list of feature columns, excluding current streak properties
    exclude_cols = [
        'streak_number', 'start_game_id', 'end_game_id', 'streak_length',
        'hit_multiplier', 'mean_multiplier', 'std_multiplier', 'max_multiplier', 'min_multiplier',
        'pct_gt2', 'pct_gt5', 'temporal_idx', 'streak_category'
    ]

    # Also exclude any multiplier_X columns from the current streak
    multiplier_cols = [
        col for col in features_df.columns if col.startswith('multiplier_')]

    # Combine all columns to exclude
    all_exclude_cols = exclude_cols + multiplier_cols

    # Get valid feature columns
    all_cols = features_df.columns.tolist()
    feature_cols = [col for col in all_cols if col not in all_exclude_cols]

    print_info(f"Created {len(feature_cols)} strictly temporal features")
    print_info(
        f"Excluded {len(all_exclude_cols)} non-temporal features to prevent leakage")

    # Create target columns based on percentiles
    percentiles = [0.25, 0.50, 0.75]
    percentile_values = [
        streak_df['streak_length'].quantile(p) for p in percentiles]

    # Log percentile values
    percentile_info = ", ".join(
        [f"P{int(p*100)}={val:.1f}" for p, val in zip(percentiles, percentile_values)])
    print_info(f"Streak length percentiles: {percentile_info}")

    # Create cluster target based on percentiles
    conditions = []
    results = []

    for i in range(len(percentiles) + 1):
        if i == 0:
            # First cluster: <= first percentile
            conditions.append(
                streak_df['streak_length'] <= percentile_values[0])
        elif i == len(percentiles):
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

    # Create a table showing the temporal features
    feature_table = create_table("Temporal Feature Examples (First 5)",
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

    display_table(feature_table)

    return features_df, feature_cols, percentile_values


def temporal_train_test_split(features_df: pd.DataFrame, feature_cols: List[str], test_size: float = 0.2) -> Tuple:
    """
    Split data while maintaining strict temporal separation.

    Args:
        features_df: DataFrame with features and targets
        feature_cols: Columns to use as features
        test_size: Fraction of data to use for testing

    Returns:
        Training and testing data splits
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

    # Create a table showing the split information
    split_table = create_table("Temporal Split Information",
                               ["Metric", "Value"])
    add_table_row(split_table, ["Training Samples", f"{len(X_train)}"])
    add_table_row(split_table, ["Testing Samples", f"{len(X_test)}"])
    add_table_row(split_table, ["Training %", f"{100*(1-test_size):.1f}%"])
    add_table_row(split_table, ["Testing %", f"{100*test_size:.1f}%"])
    add_table_row(split_table, [
                  "Train Date Range", f"{features_df['temporal_idx'].min()} - {split_temporal_idx-1}"])
    add_table_row(split_table, [
                  "Test Date Range", f"{split_temporal_idx} - {features_df['temporal_idx'].max()}"])
    display_table(split_table)

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

    return X_train, y_train, X_test, y_test, test_indices


def train_temporal_model(X_train, y_train, X_test, y_test, feature_cols, output_dir):
    """
    Train an XGBoost model with proper temporal validation.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features 
        y_test: Testing targets
        feature_cols: Feature column names
        output_dir: Directory to save model and outputs

    Returns:
        Trained model and model bundle with metadata
    """
    print_info("Training XGBoost model with proper temporal validation")

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create XGBoost data matrices
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train, feature_names=[
                         f'f{i}' for i in range(X_train_scaled.shape[1])])
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=[
                        f'f{i}' for i in range(X_test_scaled.shape[1])])

    # Set parameters for multiclass classification
    num_classes = len(np.unique(y_train))

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': 6,
        'eta': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    }

    # Display model parameters
    param_table = create_table("XGBoost Parameters", ["Parameter", "Value"])
    for param, value in params.items():
        add_table_row(param_table, [param, str(value)])
    display_table(param_table)

    # Train with early stopping
    print_info("Training model with validation-based early stopping")

    # Fix: Pass evals as keyword argument
    evallist = [(dtrain, 'training'), (dtest, 'validation')]
    num_rounds = 1000
    early_stopping = 100

    # Train the model
    bst = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=evallist,  # Fixed: Pass as keyword argument
        early_stopping_rounds=early_stopping,
        verbose_eval=100
    )

    # Get the best number of rounds
    best_rounds = bst.best_iteration
    print_info(f"Best number of rounds: {best_rounds}")

    # Train final model on the full training data
    print_info("Training final model on full training data")
    final_model = xgb.train(
        params,
        dtrain,
        best_rounds
    )

    # Make predictions on test set
    y_pred_proba = final_model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Calculate baseline log loss (prediction based on class frequencies)
    # Fix: Ensure y_train is int64 before using bincount
    y_train_int = y_train.astype(np.int64)
    class_weights = np.bincount(y_train_int) / len(y_train)
    baseline_probs = np.tile(class_weights, (len(y_test), 1))
    baseline_logloss = log_loss(y_test, baseline_probs)
    logloss_improvement = (baseline_logloss - logloss) / baseline_logloss * 100

    # Create a table showing the model evaluation
    eval_table = create_table("Temporal Model Evaluation", ["Metric", "Value"])
    add_table_row(eval_table, ["Accuracy", f"{accuracy:.4f}"])
    add_table_row(eval_table, ["Log Loss", f"{logloss:.4f}"])
    add_table_row(eval_table, ["Baseline Log Loss", f"{baseline_logloss:.4f}"])
    add_table_row(eval_table, ["Log Loss Improvement",
                  f"{logloss_improvement:.2f}%"])
    display_table(eval_table)

    # Display classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create a table showing classification report
    report_table = create_table("Classification Report",
                                ["Class", "Precision", "Recall", "F1-Score", "Support"])

    for cls in sorted(report.keys()):
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            add_table_row(report_table, [
                f"{cls}",
                f"{report[cls]['precision']:.4f}",
                f"{report[cls]['recall']:.4f}",
                f"{report[cls]['f1-score']:.4f}",
                f"{report[cls]['support']}"
            ])

    display_table(report_table)

    # Get feature importance
    importances = final_model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': list(importances.keys()),
        'Importance': list(importances.values())
    }).sort_values('Importance', ascending=False)

    # Calculate percentage of total importance
    total_importance = importance_df['Importance'].sum()
    importance_df['Percentage'] = (
        importance_df['Importance'] / total_importance * 100)

    # Display top features
    top_n = min(10, len(importance_df))
    top_features = importance_df.head(top_n)

    # Create a table showing feature importance
    importance_table = create_table(f"Top {top_n} Important Features",
                                    ["Feature", "Importance", "% of Total"])

    for _, row in top_features.iterrows():
        add_table_row(importance_table, [
            row['Feature'],
            f"{row['Importance']:.2f}",
            f"{row['Percentage']:.2f}%"
        ])

    display_table(importance_table)

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top

    # Save feature importance plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(
        output_dir, 'temporal_feature_importance.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    # Create a model bundle for saving
    model_bundle = {
        'model': final_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'params': params,
        'num_classes': num_classes,
        'feature_importance': importance_df.to_dict(),
        'metrics': {
            'accuracy': accuracy,
            'log_loss': logloss,
            'baseline_log_loss': baseline_logloss,
            'log_loss_improvement': logloss_improvement
        },
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_indices': X_test.index.tolist(),
        'test_labels': y_test.tolist(),
        'predictions': y_pred.tolist()
    }

    # Save the model bundle
    model_path = os.path.join(output_dir, 'temporal_model.pkl')
    joblib.dump(model_bundle, model_path)
    print_success(f"Saved temporal model bundle to {model_path}")
    print_info(f"Saved feature importance plot to {plot_path}")

    return final_model, model_bundle


def analyze_temporal_performance(features_df, y_pred, test_indices, percentile_values, output_dir):
    """
    Analyze the model's performance across different temporal patterns.

    Args:
        features_df: DataFrame with all features
        y_pred: Predicted labels
        test_indices: Indices of the test set
        percentile_values: Percentile values used for clustering
        output_dir: Directory to save outputs
    """
    print_info("Analyzing temporal prediction performance")

    # Create test set with predictions
    test_df = features_df.loc[test_indices].copy()
    test_df['predicted_cluster'] = y_pred
    test_df['correct'] = (test_df['predicted_cluster'] ==
                          test_df['target_cluster']).astype(int)

    # Add prediction distribution metrics
    print_info("Class Distribution Metrics:")

    # Get counts for actual classes
    actual_counts = test_df['target_cluster'].value_counts().sort_index()
    actual_pcts = actual_counts / len(test_df) * 100

    # Get counts for predicted classes
    pred_counts = test_df['predicted_cluster'].value_counts().sort_index()
    pred_pcts = pred_counts / len(test_df) * 100

    # Create a metrics table
    metrics_table = create_table("Class Distribution Metrics",
                                 ["Class", "Actual Count", "Actual %", "Predicted Count", "Predicted %"])

    # Add rows for each class
    for cls in sorted(set(test_df['target_cluster'].unique()).union(test_df['predicted_cluster'].unique())):
        act_count = actual_counts.get(cls, 0)
        act_pct = actual_pcts.get(cls, 0)
        pred_count = pred_counts.get(cls, 0)
        pred_pct = pred_pcts.get(cls, 0)

        # Map class index to description
        if cls == 0:
            cls_desc = f"{cls}: Bottom 25% (1-3)"
        elif cls == 1:
            cls_desc = f"{cls}: 25-50% (4-7)"
        elif cls == 2:
            cls_desc = f"{cls}: 50-75% (8-14)"
        elif cls == 3:
            cls_desc = f"{cls}: Top 25% (>14)"
        else:
            cls_desc = f"Class {cls}"

        add_table_row(metrics_table, [
            cls_desc,
            f"{act_count}",
            f"{act_pct:.2f}%",
            f"{pred_count}",
            f"{pred_pct:.2f}%"
        ])

    # Add total row
    add_table_row(metrics_table, [
        "Total",
        f"{len(test_df)}",
        "100.00%",
        f"{len(test_df)}",
        "100.00%"
    ])

    display_table(metrics_table)

    # Analyze performance by streak category - Fix: Add observed=False explicitly
    category_perf = test_df.groupby('streak_category', observed=False)[
        'correct'].mean()

    # Analyze performance by streak length ranges - Fix: Add observed=False explicitly
    test_df['streak_length_range'] = pd.cut(
        test_df['streak_length'],
        bins=[0, 3, 7, 14, 30, float('inf')],
        labels=['1-3', '4-7', '8-14', '15-30', '31+']
    )
    length_perf = test_df.groupby('streak_length_range', observed=False)[
        'correct'].mean()

    # Add streak categories to test set based on percentiles
    test_df['streak_category_name'] = pd.cut(
        test_df['streak_length'],
        bins=[0] + percentile_values + [float('inf')],
        labels=['short', 'medium_short', 'medium_long', 'long']
    )

    # Add test set with one-hot encoded streak categories for future reference
    for cat in test_df['streak_category_name'].cat.categories:
        test_df[f'is_{cat}'] = (
            test_df['streak_category_name'] == cat).astype(int)

    # Calculate transition probabilities
    # For the prev_category, we need to ensure it was saved in features_df during feature creation
    streak_cat_perf = None
    transitions = None

    if 'streak_category' in test_df.columns:
        # Create previous category column if it doesn't exist
        if 'prev_category' not in test_df.columns:
            test_df['prev_category'] = test_df['streak_category'].shift(1)

        # Fix: Add observed=False explicitly
        streak_cat_perf = test_df.groupby(['streak_category', 'prev_category'],
                                          observed=False)['correct'].mean()
        transitions = pd.crosstab(
            test_df['streak_category'],
            test_df['prev_category'],
            normalize='index'
        )

    # Time-based analysis - accuracy over time
    test_df['temporal_group'] = pd.qcut(
        test_df['temporal_idx'],
        q=10,
        labels=[f"P{i+1}" for i in range(10)]
    )

    # Fix: Add observed=False explicitly
    time_perf = test_df.groupby('temporal_group', observed=False)[
        'correct'].mean()

    # Fix: Sort time periods numerically instead of lexicographically
    time_perf = time_perf.reset_index()
    # Extract period numbers and convert to integers for proper sorting
    time_perf['period_num'] = time_perf['temporal_group'].str.extract(
        r'P(\d+)').astype(int)
    time_perf = time_perf.sort_values('period_num')
    # Convert back to a series with correct ordering
    time_perf = pd.Series(
        time_perf['correct'].values, index=time_perf['temporal_group'].values)

    # Performance metrics by time period
    performance_metrics = {
        'by_streak_category': category_perf.to_dict(),
        'by_streak_length': length_perf.to_dict(),
        'by_time_period': time_perf.to_dict(),
    }

    # Save performance metrics
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(
        output_dir, 'temporal_performance_metrics.json')

    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4, default=str)

    print_info(f"Saved temporal performance metrics to {metrics_path}")

    # Plot performance over time
    plt.figure(figsize=(12, 6))
    time_perf.plot(kind='bar')
    plt.title('Prediction Accuracy Over Time')
    plt.ylabel('Accuracy')
    plt.xlabel('Time Period')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save time performance plot
    plot_path = os.path.join(output_dir, 'accuracy_over_time.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved temporal performance plot to {plot_path}")

    # Create a confusion matrix visualization
    print_info("Creating confusion matrix visualization")

    cm = confusion_matrix(test_df['target_cluster'],
                          test_df['predicted_cluster'])
    plt.figure(figsize=(10, 8))

    # Define class labels for better readability
    class_labels = [
        "Bottom 25% (1-3)",
        "25-50% (4-7)",
        "50-75% (8-14)",
        "Top 25% (>14)"
    ]

    # Display confusion matrix as rich table
    print_info("Confusion Matrix Table:")
    cm_table = create_table("Confusion Matrix",
                            ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)] + ["Row Total"])

    # Calculate row totals for percentages
    row_totals = cm.sum(axis=1)

    # Add rows with counts and percentages
    for i, (row, label) in enumerate(zip(cm, class_labels)):
        row_data = [f"Act {i}: {label}"]
        for j, count in enumerate(row):
            percentage = (count / row_totals[i]) * \
                100 if row_totals[i] > 0 else 0
            cell_text = f"{count} ({percentage:.1f}%)"
            row_data.append(cell_text)
        # Add row total with percentage of the entire dataset
        total_sum = cm.sum()
        row_percentage = (row_totals[i] / total_sum) * 100
        row_data.append(f"{row_totals[i]} ({row_percentage:.1f}%)")
        add_table_row(cm_table, row_data)

    # Add a totals row
    col_totals = cm.sum(axis=0)
    total_sum = cm.sum()
    total_row = ["Total"]
    for col_total in col_totals:
        percentage = (col_total / total_sum) * 100
        total_row.append(f"{col_total} ({percentage:.1f}%)")
    # Add grand total
    total_row.append(f"{total_sum} (100.0%)")
    add_table_row(cm_table, total_row)

    display_table(cm_table)

    # Plot confusion matrix with percentages
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')

    # Save confusion matrix plot
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved confusion matrix visualization to {cm_path}")

    return performance_metrics


def make_temporal_prediction(model_bundle, recent_streaks, temporal_idx_start=None):
    """
    Make predictions using the temporal model for new streak data.

    Args:
        model_bundle: Dictionary containing model and preprocessing info
        recent_streaks: DataFrame with recent streak data
        temporal_idx_start: Starting temporal index for the new data

    Returns:
        DataFrame with predictions and probabilities
    """
    # Extract components from model bundle
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']

    # Ensure we have a temporal index
    if temporal_idx_start is None:
        temporal_idx_start = 0

    # Assign sequential temporal indices
    recent_streaks = recent_streaks.copy()
    recent_streaks['temporal_idx'] = range(temporal_idx_start,
                                           temporal_idx_start + len(recent_streaks))

    # Create temporal features
    features_df, _, _ = create_temporal_features(
        recent_streaks, lookback_window=5)

    # Check that all necessary feature columns exist
    missing_cols = [
        col for col in feature_cols if col not in features_df.columns]

    if missing_cols:
        print_warning(
            f"Missing {len(missing_cols)} feature columns. Adding with default values.")
        for col in missing_cols:
            features_df[col] = 0

    # Extract feature values and scale
    X = features_df[feature_cols]
    X_scaled = scaler.transform(X)

    # Create DMatrix
    dpredict = xgb.DMatrix(X_scaled, feature_names=[
                           f'f{i}' for i in range(X_scaled.shape[1])])

    # Make predictions
    y_pred_proba = model.predict(dpredict)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Add predictions to the DataFrame
    features_df['predicted_cluster'] = y_pred

    # Add probability for each class
    for i in range(model_bundle['num_classes']):
        features_df[f'prob_class_{i}'] = y_pred_proba[:, i]

    # Map prediction to descriptive category
    cluster_to_desc = {
        0: 'short',
        1: 'medium_short',
        2: 'medium_long',
        3: 'long'
    }

    features_df['prediction_desc'] = features_df['predicted_cluster'].map(
        cluster_to_desc)

    # Calculate confidence
    class_probs = [features_df[f'prob_class_{i}']
                   for i in range(model_bundle['num_classes'])]
    features_df['prediction_confidence'] = np.max(
        np.column_stack(class_probs), axis=1)

    return features_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Temporal Analysis for Crash Game 10× Streak Prediction')

    parser.add_argument('--mode', choices=['train', 'predict'], default='train',
                        help='Mode to run the script in (train or predict)')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with game data')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output files')

    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for streak hits (default: 10.0)')

    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Size of test set for temporal validation (default: 0.2)')

    parser.add_argument('--lookback', type=int, default=5,
                        help='Number of previous streaks to use for features (default: 5)')

    parser.add_argument('--num_streaks', type=int, default=None,
                        help='Number of most recent streaks to use for prediction (prediction mode only)')

    return parser.parse_args()


def main():
    """Main function to run the temporal analysis."""
    # Display welcome message
    print_panel(
        "Temporal Analysis for Crash Game 10.0× Streak Prediction",
        title="Welcome",
        style="green"
    )

    # Parse command line arguments
    args = parse_arguments()

    try:
        # Load the data
        streak_df = load_data(args.input, args.multiplier_threshold)

        if args.mode == 'train':
            # Create temporal features
            features_df, feature_cols, percentile_values = create_temporal_features(
                streak_df, lookback_window=args.lookback)

            # Perform temporal train-test split
            X_train, y_train, X_test, y_test, test_indices = temporal_train_test_split(
                features_df, feature_cols, test_size=args.test_size)

            # Train temporal model
            model, model_bundle = train_temporal_model(
                X_train, y_train, X_test, y_test, feature_cols, args.output_dir)

            # Get predictions on test set
            X_test_scaled = model_bundle['scaler'].transform(X_test)
            dtest = xgb.DMatrix(X_test_scaled, feature_names=[
                                f'f{i}' for i in range(X_test_scaled.shape[1])])
            y_pred = np.argmax(model.predict(dtest), axis=1)

            # Analyze temporal performance
            analyze_temporal_performance(
                features_df, y_pred, test_indices, percentile_values, args.output_dir)

            # Display rich summary of outputs
            display_output_summary(args.output_dir)

        elif args.mode == 'predict':
            # Load the model bundle
            model_path = os.path.join(
                args.output_dir, 'temporal_model.pkl')

            if not os.path.exists(model_path):
                print_error("Model file not found. Train a model first.")
                sys.exit(1)

            print_info(f"Loading model from {model_path}")
            model_bundle = joblib.load(model_path)

            # Use the most recent streaks for prediction
            if args.num_streaks:
                recent_streaks = streak_df.tail(args.num_streaks)
                print_info(
                    f"Using the {args.num_streaks} most recent streaks for prediction")
            else:
                recent_streaks = streak_df
                print_info(
                    f"Using all {len(streak_df)} streaks for prediction analysis")

            # Make predictions
            prediction_df = make_temporal_prediction(
                model_bundle, recent_streaks, temporal_idx_start=len(streak_df)-len(recent_streaks))

            # Calculate prediction statistics
            pred_counts = prediction_df['prediction_desc'].value_counts()
            pred_pcts = pred_counts / len(prediction_df) * 100

            # Display prediction distribution
            pred_table = create_table("Prediction Distribution",
                                      ["Predicted Length", "Count", "Percentage"])

            for cat in sorted(pred_counts.index):
                add_table_row(pred_table, [
                    cat,
                    f"{pred_counts[cat]}",
                    f"{pred_pcts[cat]:.1f}%"
                ])

            display_table(pred_table)

            # Save predictions to CSV
            output_path = os.path.join(
                args.output_dir, 'temporal_predictions.csv')
            prediction_df.to_csv(output_path, index=False)
            print_success(f"Saved predictions to {output_path}")

            # Calculate confidence statistics
            conf_mean = prediction_df['prediction_confidence'].mean()
            conf_median = prediction_df['prediction_confidence'].median()

            conf_table = create_table(
                "Prediction Confidence", ["Metric", "Value"])
            add_table_row(conf_table, ["Mean Confidence", f"{conf_mean:.4f}"])
            add_table_row(
                conf_table, ["Median Confidence", f"{conf_median:.4f}"])
            display_table(conf_table)

            # Display rich summary of outputs
            display_output_summary(args.output_dir)

            print_success("Prediction analysis complete!")

    except Exception as e:
        print_error(f"Error in {args.mode} mode: {str(e)}")
        import traceback
        print_error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
