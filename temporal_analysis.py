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
from imblearn.over_sampling import SMOTE  # Added for synthetic samples

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

            # New feature: rolling trend of hit multipliers
            features_df[f'hit_mult_mean_{window}'] = streak_df['hit_multiplier'].shift(
                1).rolling(window, min_periods=1).mean()
            features_df[f'hit_mult_trend_{window}'] = features_df[f'hit_mult_mean_{window}'] - \
                streak_df['hit_multiplier'].shift(window+1)

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


def train_temporal_model(X_train, y_train, X_test, y_test, feature_cols, output_dir,
                         use_class_weights=False, weight_scale=1.1, use_smote=False, smote_k_neighbors=5,
                         max_depth=6, eta=0.05, num_rounds=1000, early_stopping=100,
                         gamma=0, min_child_weight=1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8):
    """
    Train an XGBoost model with proper temporal validation.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Testing features 
        y_test: Testing targets
        feature_cols: Feature column names
        output_dir: Directory to save model and outputs
        use_class_weights: Whether to use class weights to improve recall for minority classes
        weight_scale: Scale factor for class weights (higher values give more weight to minority classes)
        use_smote: Whether to use SMOTE to generate synthetic examples
        smote_k_neighbors: Number of nearest neighbors to use for SMOTE
        max_depth: Maximum depth of XGBoost trees
        eta: Learning rate for XGBoost
        num_rounds: Maximum number of boosting rounds
        early_stopping: Early stopping rounds
        gamma: Minimum loss reduction for further partition
        min_child_weight: Minimum sum of instance weight in child
        reg_lambda: L2 regularization
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns per tree

    Returns:
        Trained model and model bundle with metadata
    """
    print_info("Training XGBoost model with proper temporal validation")

    # Apply SMOTE if requested (before scaling)
    if use_smote:
        print_info(
            f"Applying SMOTE to generate synthetic examples (k_neighbors={smote_k_neighbors})")
        original_shape = X_train.shape
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train)

        # Display resampling stats
        original_counts = pd.Series(y_train).value_counts().sort_index()
        resampled_counts = pd.Series(
            y_train_resampled).value_counts().sort_index()

        smote_table = create_table("SMOTE Resampling Results",
                                   ["Class", "Original Count", "Resampled Count", "Change"])

        for cls in sorted(resampled_counts.index):
            orig = original_counts.get(cls, 0)
            new = resampled_counts.get(cls, 0)
            change = ((new - orig) / orig * 100) if orig > 0 else float('inf')

            add_table_row(smote_table, [
                f"Class {cls}",
                f"{orig}",
                f"{new}",
                f"{change:+.1f}%"
            ])

        add_table_row(smote_table, [
            "Total",
            f"{len(y_train)}",
            f"{len(y_train_resampled)}",
            f"{(len(y_train_resampled) - len(y_train)) / len(y_train) * 100:+.1f}%"
        ])

        display_table(smote_table)

        # Update training data
        X_train = X_train_resampled
        y_train = y_train_resampled

        print_info(
            f"Reshaped training data from {original_shape} to {X_train.shape}")

    # Scale features for better model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Prepare class weights if requested
    sample_weights = None
    class_weight_dict = None

    if use_class_weights:
        print_info(f"Using class weights with scale factor {weight_scale}")

        # Count class frequencies
        class_counts = np.bincount(y_train.astype(np.int64))
        total_samples = len(y_train)
        n_classes = len(class_counts)

        # Calculate inverse frequency weights
        class_weight_dict = {}
        class_freqs = class_counts / total_samples

        for i in range(n_classes):
            # Use inverse frequency weighting adjusted by scale factor
            # Higher scale means more weight to rare classes
            class_weight = (1 / class_freqs[i]) * weight_scale
            class_weight_dict[i] = class_weight

        # No normalization step - this allows weight_scale to have direct impact
        # Note: Previously we normalized weights to average 1, which neutralized weight_scale

        # Create sample weights for each training instance
        sample_weights = np.array([class_weight_dict[y] for y in y_train])

        # Display class weights
        weight_table = create_table(
            "Class Weights", ["Class", "Count", "Frequency", "Weight"])
        for cls, weight in sorted(class_weight_dict.items()):
            count = class_counts[cls]
            freq = class_freqs[cls]
            add_table_row(weight_table, [
                f"Class {cls}",
                f"{count}",
                f"{freq:.4f}",
                f"{weight:.4f}"
            ])
        display_table(weight_table)

    # Create XGBoost data matrices
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train,
                         weight=sample_weights,  # Add sample weights if using class weights
                         feature_names=[f'f{i}' for i in range(X_train_scaled.shape[1])])
    dtest = xgb.DMatrix(X_test_scaled, label=y_test,
                        feature_names=[f'f{i}' for i in range(X_test_scaled.shape[1])])

    # Set parameters for multiclass classification
    num_classes = len(np.unique(y_train))

    params = {
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'max_depth': max_depth,
        'eta': eta,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'lambda': reg_lambda,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
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

    # Pass evals as keyword argument
    evallist = [(dtrain, 'training'), (dtest, 'validation')]

    # Train the model
    bst = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=evallist,
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

    # Map numeric classes to descriptive labels
    class_descriptions = {
        '0': "Short (1-3)",
        '1': "Medium-Short (4-7)",
        '2': "Medium-Long (8-14)",
        '3': "Long (>14)"
    }

    # Add rows for each class
    for cls in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        # Convert numeric class to string for dictionary lookup
        cls_key = str(int(float(cls))) if cls.replace(
            '.', '', 1).isdigit() else cls
        cls_desc = class_descriptions.get(cls_key, f"Class {cls}")
        add_table_row(report_table, [
            f"{cls_desc}",
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

    # Print feature name mappings
    print_info("Feature ID to Feature Name Mapping:")
    feature_map_table = create_table("Feature Name Mapping", [
                                     "Feature ID", "Feature Name"])
    for i, col in enumerate(feature_cols):
        add_table_row(feature_map_table, [f"f{i}", col])
    display_table(feature_map_table)

    # Display top features
    top_n = min(10, len(importance_df))
    top_features = importance_df.head(top_n)

    # Create a table showing feature importance
    importance_table = create_table(f"Top {top_n} Important Features",
                                    ["Feature", "Importance", "% of Total"])

    for _, row in top_features.iterrows():
        # Try to get the real feature name from feature mapping
        feature_id = row['Feature']
        feature_idx = int(feature_id.replace('f', '')
                          ) if feature_id.startswith('f') else None
        feature_name = feature_cols[feature_idx] if feature_idx is not None and feature_idx < len(
            feature_cols) else feature_id

        add_table_row(importance_table, [
            f"{feature_id} ({feature_name})",
            f"{row['Importance']:.2f}",
            f"{row['Percentage']:.2f}%"
        ])

    display_table(importance_table)

    # Plot feature importance with real names
    plt.figure(figsize=(12, 8))
    importance_with_names = top_features.copy()
    importance_with_names['DisplayName'] = importance_with_names['Feature'].apply(
        lambda f: f"{f} ({feature_cols[int(f.replace('f', ''))]})" if f.startswith('f') else f)
    plt.barh(importance_with_names['DisplayName'],
             importance_with_names['Importance'])
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
        'predictions': y_pred.tolist(),
        'class_weights': class_weight_dict,
        'used_smote': use_smote,
        'smote_params': {'k_neighbors': smote_k_neighbors} if use_smote else None
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
                            ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

    # Calculate column totals for prediction accuracy percentages
    col_totals = cm.sum(axis=0)
    total_sum = cm.sum()

    # Add rows with counts and percentages
    for i, (row, label) in enumerate(zip(cm, class_labels)):
        row_data = [f"Act {i}: {label}"]
        for j, count in enumerate(row):
            if i == j:  # Diagonal element - show prediction accuracy
                prediction_accuracy = (
                    count / col_totals[j]) * 100 if col_totals[j] > 0 else 0
                cell_text = f"{count} ({prediction_accuracy:.1f}% correct)"
            else:
                cell_text = f"{count}"
            row_data.append(cell_text)
        add_table_row(cm_table, row_data)

    # Add a totals row
    total_row = ["Total Predictions"]
    for col_total in col_totals:
        percentage = (col_total / total_sum) * 100
        total_row.append(f"{col_total} ({percentage:.1f}%)")
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


def analyze_recall_improvements(y_test, y_pred, output_dir):
    """
    Analyze and visualize recall improvements, especially for long streaks.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_dir: Directory to save outputs
    """
    print_info("Analyzing recall improvements by class")

    # Calculate precision, recall, and F1 for each class
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract metrics for each class
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for cls in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        classes.append(f"Class {cls}")
        precisions.append(report[cls]['precision'])
        recalls.append(report[cls]['recall'])
        f1_scores.append(report[cls]['f1-score'])
        supports.append(report[cls]['support'])

    # Calculate distribution metrics
    actual_counts = pd.Series(y_test).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    # Calculate percentages
    total_actual = len(y_test)
    total_pred = len(y_pred)
    actual_pcts = actual_counts / total_actual * 100
    pred_pcts = pred_counts / total_pred * 100

    # Check for prediction skew
    skew_table = create_table("Prediction Balance Check",
                              ["Class", "Actual %", "Predicted %", "Difference", "Status"])

    has_skew = False
    for cls in sorted(set(actual_counts.index.union(pred_counts.index))):
        act_pct = actual_pcts.get(cls, 0)
        pred_pct = pred_pcts.get(cls, 0)
        diff = pred_pct - act_pct
        abs_diff = abs(diff)

        # Determine status
        if abs_diff < 5:
            status = "✓ Good"
        elif abs_diff < 10:
            status = "⚠ Fair"
        else:
            status = "✗ Skewed"
            has_skew = True

        add_table_row(skew_table, [
            f"Class {cls}",
            f"{act_pct:.1f}%",
            f"{pred_pct:.1f}%",
            f"{diff:+.1f}%",
            status
        ])

    display_table(skew_table)

    if has_skew:
        print_warning(
            "Model predictions show class imbalance compared to actual distribution.")
        print_info(
            "Consider adjusting class weights or using a different balancing method.")
    else:
        print_success(
            "Model predictions match the actual class distribution well.")

    # Create a comparison bar chart
    plt.figure(figsize=(12, 8))
    width = 0.25
    x = np.arange(len(classes))

    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1-Score')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xticks(x, classes)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels
    for i, v in enumerate(precisions):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(f1_scores):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center')

    # Add class descriptions at the bottom
    class_descriptions = [
        "Short (1-3)",
        "Medium-Short (4-7)",
        "Medium-Long (8-14)",
        "Long (>14)"
    ]

    for i, desc in enumerate(class_descriptions):
        plt.text(i, -0.05, desc, ha='center', fontsize=9)

    # Add the actual vs predicted class distribution
    distribution_ax = plt.axes([0.15, 0.15, 0.3, 0.2])  # inset axes
    dist_x = np.arange(len(classes))
    dist_width = 0.35

    # Get actual and predicted percentages
    act_pct_values = []
    pred_pct_values = []

    for cls in range(len(classes)):
        act_pct_values.append(actual_pcts.get(float(cls), 0))
        pred_pct_values.append(pred_pcts.get(float(cls), 0))

    distribution_ax.bar(dist_x - dist_width/2, act_pct_values,
                        dist_width, label='Actual %')
    distribution_ax.bar(dist_x + dist_width/2, pred_pct_values,
                        dist_width, label='Predicted %')
    distribution_ax.set_title('Class Distribution')
    distribution_ax.set_ylim(
        0, max(max(act_pct_values), max(pred_pct_values)) * 1.2)
    distribution_ax.set_xticks(dist_x)
    distribution_ax.set_xticklabels([f"{i}" for i in range(len(classes))])
    distribution_ax.legend(fontsize='small')

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'recall_improvements.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved recall improvement visualization to {plot_path}")

    # Create a summary table of key findings
    insights_table = create_table("Key Recall Insights", [
                                  "Class", "Metric", "Value", "Insight"])

    # Find the class with the highest recall
    best_recall_idx = np.argmax(recalls)
    add_table_row(insights_table, [
        f"Class {best_recall_idx}",
        "Recall",
        f"{recalls[best_recall_idx]:.4f}",
        f"Best recall achieved for {class_descriptions[best_recall_idx]}"
    ])

    # Find the class with the lowest recall
    worst_recall_idx = np.argmin(recalls)
    add_table_row(insights_table, [
        f"Class {worst_recall_idx}",
        "Recall",
        f"{recalls[worst_recall_idx]:.4f}",
        f"Lowest recall for {class_descriptions[worst_recall_idx]}"
    ])

    # Find extreme precision values
    high_prec_idx = np.argmax(precisions)
    if precisions[high_prec_idx] > 0.9 and recalls[high_prec_idx] < 0.4:
        add_table_row(insights_table, [
            f"Class {high_prec_idx}",
            "Prec/Recall",
            f"{precisions[high_prec_idx]:.4f}/{recalls[high_prec_idx]:.4f}",
            f"Warning: Very high precision but low recall (conservative predictions)"
        ])

    # Check for balanced predictions
    balanced_idx = np.argmin(np.abs(np.array(precisions) - np.array(recalls)))
    add_table_row(insights_table, [
        f"Class {balanced_idx}",
        "Prec/Recall",
        f"{precisions[balanced_idx]:.4f}/{recalls[balanced_idx]:.4f}",
        f"Most balanced precision-recall for {class_descriptions[balanced_idx]}"
    ])

    display_table(insights_table)

    # Show recommended next steps based on analysis
    next_steps_table = create_table("Recommended Next Steps", [
                                    "Observation", "Action"])

    # Check for imbalanced distribution
    if has_skew:
        add_table_row(next_steps_table, [
            "Prediction distribution is skewed",
            "Reduce class weight scale or try a different balancing method"
        ])

    # Check for very high precision / low recall
    if max(precisions) > 0.9 and min(recalls) < 0.3:
        add_table_row(next_steps_table, [
            "Extreme precision-recall trade-off",
            "Adjust decision threshold or try direct calibration of probabilities"
        ])

    # Check for poor F1 scores
    if min(f1_scores) < 0.4:
        worst_f1_idx = np.argmin(f1_scores)
        add_table_row(next_steps_table, [
            f"Low F1 score for Class {worst_f1_idx}",
            f"Focus on improving features predictive of {class_descriptions[worst_f1_idx]} streaks"
        ])

    display_table(next_steps_table)

    return report


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

    # Add game IDs and streak information from recent_streaks
    features_df['start_game_id'] = recent_streaks['start_game_id']
    features_df['end_game_id'] = recent_streaks['end_game_id']
    features_df['streak_number'] = recent_streaks.index
    features_df['streak_length'] = recent_streaks['streak_length']

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

    # Create a mutually exclusive group for class balancing methods
    class_balance_group = parser.add_mutually_exclusive_group()

    class_balance_group.add_argument('--use_class_weights', action='store_true',
                                     help='Use class weights to improve recall for minority classes')

    parser.add_argument('--weight_scale', type=float, default=1.1,
                        help='Scale factor for class weights (higher values give more weight to minority classes)')

    class_balance_group.add_argument('--use_smote', action='store_true',
                                     help='Use SMOTE to generate synthetic examples for minority classes')

    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                        help='Number of nearest neighbors to use for SMOTE (default: 5)')

    # Model parameters
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum depth of XGBoost trees (default: 6)')

    # New hyperparameters for regularization
    parser.add_argument('--eta', type=float, default=0.05,
                        help='Learning rate for XGBoost (default: 0.05, recommended range: 0.03-0.05)')

    parser.add_argument('--num_rounds', type=int, default=1000,
                        help='Maximum number of boosting rounds (default: 1000)')

    parser.add_argument('--early_stopping', type=int, default=100,
                        help='Early stopping rounds (default: 100)')

    parser.add_argument('--gamma', type=float, default=0,
                        help='Minimum loss reduction for further partition (default: 0, recommended for tuning: 0.5-1.0)')

    parser.add_argument('--min_child_weight', type=float, default=1,
                        help='Minimum sum of instance weight in child (default: 1, recommended for tuning: 1-5)')

    parser.add_argument('--reg_lambda', type=float, default=1.0,
                        help='L2 regularization (default: 1.0, recommended for tuning: 1.0-1.5)')

    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio of training data (default: 0.8)')

    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Subsample ratio of columns per tree (default: 0.8)')

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

    # Double check - warn if both methods are somehow selected
    if args.use_class_weights and args.use_smote:
        print_warning(
            "WARNING: Both class weights and SMOTE are enabled. This can cause extreme prediction bias.")
        print_warning("Consider using only one class balancing method.")
        print_info("Continuing with both methods enabled as requested...")

    try:
        # Load the data
        streak_df = load_data(args.input, args.multiplier_threshold)

        if args.mode == 'train':
            # Create temporal features
            features_df, feature_cols, percentile_values = create_temporal_features(
                streak_df, lookback_window=args.lookback)

            # Print total feature count and newly added features
            new_features = [
                col for col in feature_cols
                if col.startswith('hit_mult_') or col.startswith('time_since_')
            ]

            if len(new_features) > 0:
                print_info(
                    f"Added {len(new_features)} new features: {', '.join(new_features[:5])}...")

            # Display model configuration
            print_panel(
                f"Model Configuration:\n"
                f"- Max Tree Depth: {args.max_depth}\n"
                f"- Learning Rate (eta): {args.eta}\n"
                f"- Max Rounds: {args.num_rounds} (early stop: {args.early_stopping})\n"
                f"- Regularization: gamma={args.gamma}, lambda={args.reg_lambda}, min_child_weight={args.min_child_weight}\n"
                f"- Sampling: subsample={args.subsample}, colsample_bytree={args.colsample_bytree}\n"
                f"- Class Weights: {'Yes' if args.use_class_weights else 'No'}\n"
                f"- Weight Scale: {args.weight_scale if args.use_class_weights else 'N/A'}\n"
                f"- SMOTE: {'Yes' if args.use_smote else 'No'}\n"
                f"- Features: {len(feature_cols)} temporal features",
                title="Configuration",
                style="cyan"
            )

            # Perform temporal train-test split
            X_train, y_train, X_test, y_test, test_indices = temporal_train_test_split(
                features_df, feature_cols, test_size=args.test_size)

            # Print class balancing strategy information
            if args.use_class_weights:
                print_info(
                    f"Class balancing strategy: Using class weights (scale={args.weight_scale})")
            elif args.use_smote:
                print_info(
                    f"Class balancing strategy: Using SMOTE (k_neighbors={args.smote_k_neighbors})")
            else:
                print_info(
                    "Class balancing strategy: None (using original class distribution)")

            # Train temporal model
            model, model_bundle = train_temporal_model(
                X_train, y_train, X_test, y_test, feature_cols, args.output_dir,
                use_class_weights=args.use_class_weights,
                weight_scale=args.weight_scale,
                use_smote=args.use_smote,
                smote_k_neighbors=args.smote_k_neighbors,
                max_depth=args.max_depth,
                eta=args.eta,
                num_rounds=args.num_rounds,
                early_stopping=args.early_stopping,
                gamma=args.gamma,
                min_child_weight=args.min_child_weight,
                reg_lambda=args.reg_lambda,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree)

            # Get predictions on test set
            X_test_scaled = model_bundle['scaler'].transform(X_test)
            dtest = xgb.DMatrix(X_test_scaled, feature_names=[
                                f'f{i}' for i in range(X_test_scaled.shape[1])])
            y_pred = np.argmax(model.predict(dtest), axis=1)

            # Analyze recall improvements
            if args.use_class_weights or args.use_smote:
                report = analyze_recall_improvements(
                    y_test, y_pred, args.output_dir)

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

            # Display sample predictions with game IDs and streak information
            sample_table = create_table("Sample Predictions",
                                        ["Streak #", "Start Game ID", "End Game ID", "Streak Length", "Predicted", "Confidence"])

            # Show the last 10 predictions
            for _, row in prediction_df.tail(10).iterrows():
                add_table_row(sample_table, [
                    f"{row['streak_number']}",
                    f"{row['start_game_id']}",
                    f"{row['end_game_id']}",
                    f"{row['streak_length']}",
                    f"{row['prediction_desc']}",
                    f"{row['prediction_confidence']:.3f}"
                ])

            display_table(sample_table)

            # If ground truth is available, show confusion matrix
            if 'target_cluster' in prediction_df.columns:
                print_info("Creating confusion matrix for predictions")

                cm = confusion_matrix(prediction_df['target_cluster'],
                                      prediction_df['predicted_cluster'])

                # Define class labels for better readability
                class_labels = [
                    "Short (1-3)",
                    "Medium-Short (4-7)",
                    "Medium-Long (8-14)",
                    "Long (>14)"
                ]

                # Display confusion matrix as rich table
                cm_table = create_table("Prediction Confusion Matrix",
                                        ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

                # Calculate row totals for recall calculation
                row_totals = cm.sum(axis=1)

                # Add rows with counts and percentages
                for i, (row, label) in enumerate(zip(cm, class_labels)):
                    row_data = [f"Act {i}: {label}"]
                    for j, count in enumerate(row):
                        recall = (count / row_totals[i]) * \
                            100 if row_totals[i] > 0 else 0
                        if i == j:  # Diagonal element - show recall percentage
                            cell_text = f"{count} ({recall:.1f}% recall)"
                        else:
                            cell_text = f"{count}"
                        row_data.append(cell_text)
                    add_table_row(cm_table, row_data)

                display_table(cm_table)

                # Plot and save confusion matrix
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
                plt.title('Prediction Confusion Matrix')

                # Save confusion matrix plot
                cm_path = os.path.join(
                    args.output_dir, 'prediction_confusion_matrix.png')
                plt.savefig(cm_path, bbox_inches='tight')
                plt.close()

                print_info(f"Saved prediction confusion matrix to {cm_path}")

                # Add precision confusion matrix
                print_info("Creating precision-focused confusion matrix")

                # Calculate column totals for precision calculation
                col_totals = cm.sum(axis=0)

                # Create precision matrix table
                precision_table = create_table("Prediction Confusion Matrix (Precision)",
                                               ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

                # Add rows with counts and precision percentages
                for i, (row, label) in enumerate(zip(cm, class_labels)):
                    row_data = [f"Act {i}: {label}"]
                    for j, count in enumerate(row):
                        precision = (
                            count / col_totals[j]) * 100 if col_totals[j] > 0 else 0
                        if i == j:  # Diagonal element - show precision percentage
                            cell_text = f"{count} ({precision:.1f}% precision)"
                        else:
                            cell_text = f"{count}"
                        row_data.append(cell_text)
                    add_table_row(precision_table, row_data)

                # Add total row showing column totals
                total_row = ["Total Predictions"]
                for j, col_total in enumerate(col_totals):
                    # Diagonal element is correct predictions
                    correct = cm[j, j]
                    precision = (correct / col_total) * \
                        100 if col_total > 0 else 0
                    total_row.append(f"{col_total} ({precision:.1f}% overall)")

                add_table_row(precision_table, total_row)
                display_table(precision_table)

                # Plot and save the precision confusion matrix
                plt.figure(figsize=(10, 8))
                # Normalize by column (axis=0) to get precision
                cm_norm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
                # Replace NaN with 0 for empty columns
                cm_norm = np.nan_to_num(cm_norm)
                sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues',
                            xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel('Predicted Class')
                plt.ylabel('Actual Class')
                plt.title('Precision Confusion Matrix')

                # Save precision confusion matrix plot
                precision_cm_path = os.path.join(
                    args.output_dir, 'prediction_precision_matrix.png')
                plt.savefig(precision_cm_path, bbox_inches='tight')
                plt.close()

                print_info(
                    f"Saved precision confusion matrix to {precision_cm_path}")

                # Add classification report
                print_info("Generating classification report")
                report = classification_report(prediction_df['target_cluster'],
                                               prediction_df['predicted_cluster'],
                                               output_dict=True)

                # Create a classification report table
                report_table = create_table("Classification Report",
                                            ["Class", "Precision", "Recall", "F1-Score", "Support"])

                # Map numeric classes to descriptive labels
                class_descriptions = {
                    '0': "Short (1-3)",
                    '1': "Medium-Short (4-7)",
                    '2': "Medium-Long (8-14)",
                    '3': "Long (>14)"
                }

                # Add rows for each class
                for cls in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
                    # Convert numeric class to string for dictionary lookup
                    cls_key = str(int(float(cls))) if cls.replace(
                        '.', '', 1).isdigit() else cls
                    cls_desc = class_descriptions.get(cls_key, f"Class {cls}")
                    add_table_row(report_table, [
                        f"{cls_desc}",
                        f"{report[cls]['precision']:.4f}",
                        f"{report[cls]['recall']:.4f}",
                        f"{report[cls]['f1-score']:.4f}",
                        f"{report[cls]['support']}"
                    ])

                # Add accuracy row
                if 'accuracy' in report:
                    add_table_row(report_table, [
                        "Accuracy",
                        "",
                        "",
                        f"{report['accuracy']:.4f}",
                        f"{sum([report[cls]['support'] for cls in report if cls not in ['accuracy', 'macro avg', 'weighted avg']])}"
                    ])

                display_table(report_table)

                # Save classification report to JSON
                report_path = os.path.join(
                    args.output_dir, 'prediction_classification_report.json')
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                print_info(f"Saved classification report to {report_path}")

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

            # Display only prediction-related outputs in summary for predict mode
            output_files = []

            # Add only prediction-relevant files to the output summary
            if os.path.exists(output_path):
                output_files.append(('temporal_predictions.csv', output_path))

            if 'target_cluster' in prediction_df.columns and os.path.exists(cm_path):
                output_files.append(
                    ('prediction_confusion_matrix.png', cm_path))
                if os.path.exists(precision_cm_path):
                    output_files.append(
                        ('prediction_precision_matrix.png', precision_cm_path))
                if os.path.exists(report_path):
                    output_files.append(
                        ('prediction_classification_report.json', report_path))

            if output_files:
                from rich_summary import display_custom_output_summary
                display_custom_output_summary(
                    output_files, "Prediction Output Files")
            else:
                # Skip summary display for predict mode
                pass

            print_success("Prediction analysis complete!")

    except Exception as e:
        print_error(f"Error in {args.mode} mode: {str(e)}")
        import traceback
        print_error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
