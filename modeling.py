#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modeling module for Crash Game Streak Analysis.

This module handles model training, evaluation, and prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.metrics import log_loss, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Any
import importlib

# Import rich logging
from logger_config import (
    console, create_table, display_table, add_table_row,
    create_stats_table, print_info, print_success, print_warning,
    print_error, print_panel
)

logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, feature_cols: List[str],
                test_frac: float, random_seed: int, eval_folds: int = 5,
                output_dir: str = './output', multiplier_threshold: float = 10.0,
                percentiles: List[float] = [0.25, 0.50, 0.75], window: int = 50,
                scaler=None) -> Dict[str, Any]:
    """
    Train an XGBoost model for streak length prediction.

    Args:
        df: DataFrame with features and target from streak-based analysis
        feature_cols: List of feature column names
        test_frac: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        eval_folds: Number of folds for cross-validation
        output_dir: Directory to save model and outputs
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])
        window: Lookback window size for cross-validation
        scaler: Fitted StandardScaler object for feature normalization

    Returns:
        Model bundle dictionary containing trained model, scaler, and metadata
    """
    logger.info("Training XGBoost model for streak length prediction")

    # Import data_processing module to use the new function
    from data_processing import prepare_train_test_features

    # Check if df already has features or is raw game data
    if "Game ID" in df.columns and "Bust" in df.columns:
        # This is raw game data - we need to prepare features with proper train-test split
        logger.info(
            "Raw game data detected - preparing features with train-test split to prevent data leakage")
        train_df, test_df, feature_cols, scaler = prepare_train_test_features(
            df, window, test_frac, random_seed, multiplier_threshold, percentiles
        )
    else:
        # This is pre-processed feature data - we should warn clearly about leakage risk
        logger.warning(
            "Using pre-processed feature data - this may cause SEVERE data leakage issues")
        logger.warning(
            "Consider using raw game data with Game ID and Bust columns for proper time-based splitting")

        # Check if this is likely a feature dataframe
        if 'target_cluster' not in df.columns:
            logger.error(
                "DataFrame does not contain 'target_cluster' column. Cannot proceed with training.")
            raise ValueError(
                "DataFrame must contain 'target_cluster' column if not providing raw game data")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Use time-ordered train/test split preserving streak continuity
        if 'streak_number' in df.columns:
            logger.info(
                "Sorting by streak_number to maintain temporal ordering")
            df = df.sort_values('streak_number')
        else:
            logger.warning(
                "No streak_number column found. Cannot guarantee temporal ordering.")

        # Use sequential split by index to ensure we don't break streak continuity
        split_idx = int(len(df) * (1 - test_frac))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Show split information
        split_info = {
            "Training Streaks": len(train_df),
            "Testing Streaks": len(test_df),
            "Training Split": f"{(1-test_frac)*100:.1f}%",
            "Testing Split": f"{test_frac*100:.1f}%"
        }
        create_stats_table("Streak Train/Test Split", split_info)

    # Check for missing values in feature columns
    missing_values = train_df[feature_cols].isna().sum().sum()
    if missing_values > 0:
        print_warning(
            f"Found {missing_values} missing values in feature columns. Filling with zeros.")
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)

    X_train, y_train = train_df[feature_cols], train_df["target_cluster"]
    X_test, y_test = test_df[feature_cols], test_df["target_cluster"]

    # Calculate baseline probabilities
    unique_classes = sorted(y_train.unique())
    baseline_probs = {}
    for c in unique_classes:
        baseline_probs[c] = (y_train == c).mean()

    print_info(
        f"Baseline probabilities: {', '.join([f'Class {k}: {v:.3f}' for k, v in baseline_probs.items()])}")

    # Calculate empirical hit rate (this is now 1.0 since we're only analyzing completed streaks)
    p_hat = 1.0
    print_info(
        f"Empirical {multiplier_threshold}× hit rate in streaks: {p_hat:.4f}")

    # Display baseline probabilities
    baseline_table = create_table("Baseline Probabilities", [
                                  "Cluster", "Description", "Probability"])

    # Create dynamic cluster descriptions based on percentiles
    cluster_descriptions = {}
    # Calculate the actual streak length values at each percentile
    percentile_values = []
    if 'target_streak_length' in train_df.columns:
        percentile_values = [
            train_df['target_streak_length'].quantile(p) for p in percentiles]
    else:
        # Use an approximation if target_streak_length is not available
        logger.warning(
            "No target_streak_length column found. Using approximations for cluster descriptions.")
        # Default values based on common observations
        percentile_values = [3, 7, 14]

    for i in range(len(percentiles) + 1):
        if i == 0:
            cluster_descriptions[i] = f"Cluster {i}: Bottom {int(percentiles[0]*100)}% (1-{int(percentile_values[0])} streak length)"
        elif i == len(percentiles):
            cluster_descriptions[i] = f"Cluster {i}: Top {int((1-percentiles[-1])*100)}% (>{int(percentile_values[-1])} streak length)"
        else:
            lower = int(percentiles[i-1]*100)
            upper = int(percentiles[i]*100)
            lower_streak = int(percentile_values[i-1]) + 1
            upper_streak = int(percentile_values[i])
            cluster_descriptions[
                i] = f"Cluster {i}: {lower}-{upper} percentile ({lower_streak}-{upper_streak} streak length)"

    for cluster_id, prob in baseline_probs.items():
        cluster_desc = cluster_descriptions.get(
            cluster_id, f"Cluster {cluster_id}")
        add_table_row(baseline_table, [
                      cluster_id, cluster_desc, f"{prob*100:.2f}%"])
    display_table(baseline_table)

    baseline_pred = np.tile(list(baseline_probs.values()), (len(y_test), 1))
    logloss_baseline = log_loss(y_test, baseline_pred)
    print_info(f"Baseline geometric log-loss: {logloss_baseline:.4f}")

    # Configure XGBoost parameters
    num_classes = len(baseline_probs)
    params = dict(
        objective="multi:softprob",
        num_class=num_classes,
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        seed=random_seed
    )

    # Store lookback window size in model bundle for later use, but don't pass to XGBoost
    lookback_window = window

    # Display model parameters
    param_table = create_table("XGBoost Parameters", ["Parameter", "Value"])
    for param, value in params.items():
        add_table_row(param_table, [param, str(value)])
    display_table(param_table)

    print_info(f"Performing {eval_folds}-fold cross-validation (optimized)")

    # Create a table for CV results
    cv_table = create_table("Cross-Validation Results",
                            ["Fold", "Best Round", "Log Loss"])

    # More efficient cross-validation approach
    # Prepare DMatrix for training and testing
    dtrain_full = xgb.DMatrix(X_train, label=y_train,
                              feature_names=feature_cols)

    # Use reduced number of rounds for efficiency
    max_rounds = 2000
    early_stopping = 100

    # Set fewer rounds per fold to speed up CV
    sample_size = min(20000, len(X_train))  # Limit sample size for CV
    fold_size = sample_size // eval_folds

    # Use a smaller subsample for each fold to speed up CV
    best_ntrees = []

    # Perform CV using a sample of data for each fold
    n_train = len(X_train)
    for fold, (tr_idx, val_idx) in enumerate(rolling_origin_indices(n_train, eval_folds), 1):

        # Ensure validation set is not too large
        if len(val_idx) > 5000:
            val_idx = list(val_idx)[:5000]

        # Create DMatrix for training and validation
        dtr = xgb.DMatrix(
            X_train.iloc[tr_idx], label=y_train.iloc[tr_idx], feature_names=feature_cols)
        dval = xgb.DMatrix(
            X_train.iloc[val_idx], label=y_train.iloc[val_idx], feature_names=feature_cols)

        # Train with early stopping
        print_info(
            f"Training fold {fold} with {len(tr_idx)} training samples and {len(val_idx)} validation samples")
        bst = xgb.train(
            params, dtr,
            num_boost_round=max_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False
        )

        # Record results
        br = best_round(bst, default_rounds=max_rounds)
        best_ntrees.append(br)

        # Add row to CV table
        best_score = bst.best_score if hasattr(bst, 'best_score') else "N/A"
        add_table_row(cv_table, [fold, br, f"{best_score:.4f}" if isinstance(
            best_score, float) else best_score])

    display_table(cv_table)

    # Use median of best rounds but limit to reasonable number
    nrounds = min(max_rounds, int(np.median(best_ntrees)))
    print_info(f"Using n_rounds = {nrounds} (median from cross-validation)")

    # Final fit on all training data
    print_info("Training final model on all streak training data")
    bst_final = xgb.train(params, dtrain_full, num_boost_round=nrounds)

    # Save the model with scaler
    model_path = os.path.join(output_dir, "xgboost_model.pkl")

    # Check if we have a scaler
    if scaler is None:
        print_warning(
            "No scaler provided! Model predictions may be inconsistent.")
        print_warning(
            "Creating a new scaler to capture feature distributions...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit scaler on the features we have (not ideal, but better than nothing)
        scaler.fit(df[feature_cols])

    # Log scaler information
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        print_info(
            f"StandardScaler included in model bundle with {len(scaler.mean_)} features")
        print_info(
            f"Feature scaling mean range: [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
        print_info(
            f"Feature scaling std range: [{scaler.scale_.min():.4f}, {scaler.scale_.max():.4f}]")

    # Save model bundle to disk for later use
    # Include the scaler, feature columns, and configuration details
    model_bundle = {
        "model": bst_final,
        "scaler": scaler,  # Include the scaler for consistent scaling in prediction
        "feature_cols": feature_cols,  # Include feature columns list
        "percentiles": percentiles,
        "percentile_values": percentile_values,
        "multiplier_threshold": multiplier_threshold,
        "window": window,
        "baseline_probs": baseline_probs,
        "p_hat": p_hat,
        "model_version": "1.1",  # Increment version to indicate fixed data leakage
        "training_date": pd.Timestamp.now().isoformat(),
        "num_features": len(feature_cols),
        "num_classes": num_classes
    }

    # Save the model bundle to disk
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    joblib.dump(model_bundle, model_path)
    print_success(
        f"Saved streak-based model bundle to {model_path} (including scaler)")

    # Create DMatrix for testing
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Generate confusion matrix and evaluation metrics on the test set
    generate_confusion_matrix(
        X_test, y_test, bst_final, feature_cols, output_dir, percentiles)

    # Calculate expected calibration error
    y_probs = bst_final.predict(dtest)
    ece = expected_calibration_error(y_test, y_probs)

    # Display evaluation metrics
    eval_metrics = {
        "Log Loss (Baseline)": f"{logloss_baseline:.4f}",
        "Log Loss (Model)": f"{log_loss(y_test, y_probs):.4f}",
        "Log Loss Improvement": f"{(1 - (log_loss(y_test, y_probs) / logloss_baseline)) * 100:.2f}%",
        "Expected Calibration Error": f"{ece:.4f}"
    }
    create_stats_table("Model Evaluation Metrics", eval_metrics)

    return model_bundle


def rolling_origin_indices(n_train: int, n_folds: int, min_val_size: int = 1000, gap: int = 0):
    """
    Generate indices for time-series cross-validation using rolling origin approach.

    This approach respects the temporal nature of the data by ensuring that:
    1. Only past data is used for training
    2. A gap can be introduced between training and validation to simulate forecasting
    3. Validation sets represent future data from the training set's perspective

    Args:
        n_train: Number of training samples
        n_folds: Number of folds for cross-validation
        min_val_size: Minimum size of validation set
        gap: Number of samples to skip between training and validation sets

    Yields:
        Tuples of (train_indices, validation_indices)
    """
    # Validate inputs
    if n_train <= 0:
        raise ValueError("n_train must be positive")
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")

    # Determine validation set size - at least min_val_size
    val_size = max(min_val_size, n_train // (n_folds * 2))

    # Calculate fold sizes to ensure all data is used approximately evenly
    fold_size = (n_train - val_size) // n_folds

    # For each fold
    for i in range(n_folds):
        # Calculate how much data is available for this fold
        if i == n_folds - 1:
            # Last fold - use all remaining data
            train_end = n_train - val_size - gap
        else:
            # Regular fold
            train_end = min((i + 1) * fold_size, n_train - val_size - gap)

        # Training data is all data up to train_end
        train_indices = list(range(0, train_end))

        # Validation data starts after train_end + gap
        val_start = train_end + gap
        val_end = min(val_start + val_size, n_train)
        validation_indices = list(range(val_start, val_end))

        # Skip fold if validation set is too small
        if len(validation_indices) < min_val_size // 2:
            continue

        yield train_indices, validation_indices


def best_round(bst, default_rounds: int) -> int:
    """
    Return the best number of rounds from an XGBoost model.

    Args:
        bst: XGBoost model
        default_rounds: Default number of rounds if best not found

    Returns:
        Best number of rounds
    """
    if hasattr(bst, "best_ntree_limit") and bst.best_ntree_limit:   # XGB ≤ 1.7
        return bst.best_ntree_limit
    if hasattr(bst, "best_iteration") and bst.best_iteration:       # XGB ≥ 1.7
        return bst.best_iteration + 1  # convert 0-based → count
    return default_rounds  # no early-stop hit


def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """
    Calculate expected calibration error.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Expected calibration error
    """
    prob_true, prob_pred = calibration_curve(
        y_true == 0, y_prob[:, 0], n_bins=n_bins)  # crude binary proxy
    return np.abs(prob_true - prob_pred).mean()


def generate_confusion_matrix(X_test, y_test, model, feature_cols, output_dir,
                              percentiles: List[float] = [0.25, 0.50, 0.75]) -> None:
    """
    Generate and save confusion matrix for test set.

    Args:
        X_test: Test features
        y_test: Test labels
        model: Trained XGBoost model
        feature_cols: List of feature column names
        output_dir: Directory to save outputs
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])
    """
    # Prepare data
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    probs = model.predict(dtest)

    # Get the number of classes from the probabilities
    num_classes = probs.shape[1]

    # Create column names based on the number of classes
    prob_columns = [f"prob_class{i}" for i in range(num_classes)]

    # Assemble into a DataFrame
    results = pd.DataFrame(probs,
                           columns=prob_columns,
                           index=X_test.index)
    results["actual"] = y_test.values
    results["predicted"] = results[prob_columns].idxmax(axis=1) \
        .map({f"prob_class{i}": i for i in range(num_classes)})

    # Generate confusion matrix
    cm = confusion_matrix(results["actual"], results["predicted"])

    # Calculate the actual streak length values at each percentile
    # This should be available from the training data, here we use a simplified approach
    # In real usage, these values should be passed in or calculated from the original data
    try:
        # Get the training set to calculate percentile values
        train_df = pd.concat([X_test, y_test], axis=1)
        if hasattr(train_df, 'to_pandas'):
            train_df = train_df.to_pandas()

        # Try to find the original column with streak lengths, which might be stored in the model
        # This is a fallback approach that might not always work
        percentile_values = []
        if 'target_streak_length' in train_df.columns:
            percentile_values = [
                train_df['target_streak_length'].quantile(p) for p in percentiles]
        else:
            # Use default placeholders if we can't calculate actual streak lengths
            percentile_values = [4, 8, 15]  # Example placeholder values
    except Exception:
        # Fallback to placeholder values if something goes wrong
        percentile_values = [4, 8, 15]  # Example placeholder values

    # Create class descriptions based on percentiles with streak lengths
    class_labels = []
    for i in range(len(percentiles) + 1):
        if i == 0:
            class_labels.append(
                f"{i}: <{int(percentiles[0]*100)}% (1-{int(percentile_values[0])})")
        elif i == len(percentiles):
            class_labels.append(
                f"{i}: >{int(percentiles[-1]*100)}% (>{int(percentile_values[-1])})")
        else:
            lower = int(percentiles[i-1]*100)
            upper = int(percentiles[i]*100)
            lower_streak = int(percentile_values[i-1]) + 1
            upper_streak = int(percentile_values[i])
            class_labels.append(
                f"{i}: {lower}-{upper}% ({lower_streak}-{upper_streak})")

    # Display confusion matrix as a table
    cm_table = create_table("Confusion Matrix",
                            [""] + [f"Pred {label}" for label in class_labels])

    # Add rows for the confusion matrix
    for i in range(cm.shape[0]):
        row_data = [f"Act {class_labels[i]}"] + [str(cm[i, j])
                                                 for j in range(cm.shape[1])]
        add_table_row(cm_table, row_data)

    display_table(cm_table)

    # Add accuracy
    accuracy = np.diag(cm).sum() / cm.sum()
    print_info(f"Confusion Matrix Accuracy: {accuracy:.4f}")

    # Calculate and display per-class metrics
    cm_metrics = create_table(
        "Per-class Metrics", ["Class", "Precision", "Recall", "F1-Score"])

    for i in range(cm.shape[0]):
        # Calculate metrics
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive

        precision = true_positive / \
            (true_positive + false_positive) if (true_positive +
                                                 false_positive) > 0 else 0
        recall = true_positive / \
            (true_positive + false_negative) if (true_positive +
                                                 false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        add_table_row(
            cm_metrics, [class_labels[i], f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

    display_table(cm_metrics)

    # Save detailed results to CSV
    results_path = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(results_path)
    print_info(f"Saved detailed test predictions to {results_path}")


def predict_next_cluster(model_or_path, last_streaks: List[Dict], window: int,
                         feature_cols: List[str] = None, multiplier_threshold: float = 10.0,
                         percentiles: List[float] = [0.25, 0.50, 0.75],
                         scaler=None) -> Dict[str, float]:
    """
    Predict the next streak length cluster based on recent streak patterns.

    Args:
        model_or_path: Trained XGBoost model or path to model file
        last_streaks: List of dictionaries with recent streak information
        window: Number of previous streaks to consider
        feature_cols: List of feature column names (if not included in model bundle)
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])
        scaler: StandardScaler for feature normalization (if not included in model bundle)

    Returns:
        Dictionary of cluster probabilities
    """
    from data_processing import make_feature_vector, create_streak_features

    # Load model if path is provided
    model_bundle = None
    if isinstance(model_or_path, str):
        print_info(f"Loading model from file: {model_or_path}")
        try:
            model_bundle = joblib.load(model_or_path)
            model = model_bundle.get("model")

            # Get saved feature columns from bundle
            saved_feature_cols = model_bundle.get("feature_cols")
            if saved_feature_cols:
                print_info(
                    f"Loaded feature columns from model bundle: {len(saved_feature_cols)} features")
                feature_cols = saved_feature_cols

            # Get saved scaler from bundle if not provided
            if scaler is None and "scaler" in model_bundle:
                scaler = model_bundle.get("scaler")
                print_info(
                    "Using scaler from model bundle for feature normalization")
                if hasattr(scaler, 'mean_'):
                    print_info(f"Scaler has {len(scaler.mean_)} features")

            # Log bundle version information
            if "version" in model_bundle:
                print_info(
                    f"Model bundle version: {model_bundle.get('version')}")
        except Exception as e:
            print_error(f"Error loading model: {str(e)}")
            print_warning("Attempting to proceed with fallback approach...")
            # Try simple load as a fallback
            try:
                model = joblib.load(model_or_path)
            except Exception as e2:
                print_error(f"Fallback load also failed: {str(e2)}")
                raise ValueError(f"Could not load model from {model_or_path}")
    else:
        model = model_or_path  # Already a model object
        print_info("Using provided model object directly")

        # Check if it's actually a bundle
        if isinstance(model, dict) and "model" in model:
            print_info(
                "Model object appears to be a bundle, extracting components...")
            if "scaler" in model and scaler is None:
                scaler = model.get("scaler")
                print_info("Using scaler from model bundle")
            if "feature_cols" in model and feature_cols is None:
                feature_cols = model.get("feature_cols")
                print_info(
                    f"Using {len(feature_cols)} feature columns from model bundle")
            model = model.get("model")

    # Handle empty streaks gracefully
    if not last_streaks:
        print_warning(
            "No streaks provided for prediction. Using default probabilities.")
        # Return uniform distribution across classes
        num_classes = len(percentiles) + 1
        return {str(i): 1.0 / num_classes for i in range(num_classes)}

    # Check if we have enough streaks
    if len(last_streaks) < window:
        print_warning(
            f"Expected at least {window} streaks, but got {len(last_streaks)}. Results may be less accurate.")

    # Display detailed information about streaks used for prediction
    streaks_table = create_table("Input Streaks for Prediction",
                                 ["Streak #", "Game ID Range", "Length", "Hit Multiplier"])

    # Display streaks in reverse order (most recent first)
    for i, streak in enumerate(reversed(last_streaks[:window])):
        if 'streak_number' in streak and 'start_game_id' in streak and 'end_game_id' in streak:
            streak_num = streak.get('streak_number', 'N/A')
            start_id = streak.get('start_game_id', 'N/A')
            end_id = streak.get('end_game_id', 'N/A')
            length = streak.get('streak_length', 'N/A')
            hit_mult = streak.get('hit_multiplier', 'N/A')

            # Format as "Streak #123 (most recent)" for the most recent streak
            streak_label = f"Streak #{streak_num}"
            if i == 0:
                streak_label += " (most recent)"

            game_range = f"{start_id} → {end_id}"

            add_table_row(streaks_table, [
                streak_label,
                game_range,
                length,
                f"{hit_mult:.2f}" if isinstance(
                    hit_mult, (int, float)) else hit_mult
            ])

    # If we have a next streak id, show prediction target
    next_streak_id = None
    if len(last_streaks) > 0 and 'streak_number' in last_streaks[-1]:
        next_streak_id = last_streaks[-1]['streak_number'] + 1
        add_table_row(streaks_table, [
            f"Streak #{next_streak_id} (predicted)",
            "? → ?",
            "?",
            "?"
        ])

    display_table(streaks_table)

    try:
        # Create DataFrame from streaks
        streak_df = pd.DataFrame(last_streaks)
        print_info(f"Creating features from {len(streak_df)} streaks")

        # Log key information about streaks
        if 'streak_length' in streak_df.columns:
            print_info(f"Streak lengths: min={streak_df['streak_length'].min()}, "
                       f"max={streak_df['streak_length'].max()}, "
                       f"mean={streak_df['streak_length'].mean():.2f}")

        if not set(['mean_multiplier', 'max_multiplier', 'min_multiplier', 'streak_length']).issubset(streak_df.columns):
            print_warning(
                f"Missing expected columns in streak DataFrame. Available columns: {streak_df.columns.tolist()}")

        # For prediction, use the create_streak_features with prediction_mode=True
        from data_processing import create_streak_features
        features_df = create_streak_features(
            streak_df, lookback_window=window, prediction_mode=True)

        # Get the last row for prediction - prediction_mode keeps all rows
        if not features_df.empty:
            last_features = features_df.iloc[-1]
            print_info(
                f"Successfully created features from streak data with {len(features_df.columns)} columns")

            # Log key feature information
            if set(['mean_multiplier', 'max_multiplier', 'min_multiplier']).issubset(features_df.columns):
                print_info(f"Key features for last row: mean_mult={last_features['mean_multiplier']:.4f}, "
                           f"max_mult={last_features['max_multiplier']:.4f}, "
                           f"min_mult={last_features['min_multiplier']:.4f}")
        else:
            raise ValueError("Empty features DataFrame created")

        # Align feature vector with the expected feature names from training
        if feature_cols:
            # Create a Series with the expected feature columns, filling missing ones with 0
            aligned_features = pd.Series(0.0, index=feature_cols)

            # Update values where features exist
            feature_matches = 0
            for col in feature_cols:
                if col in last_features:
                    # Extract scalar value to avoid "setting an array element with a sequence" error
                    try:
                        val = last_features[col]
                        # Check if value is array-like and extract first element if needed
                        if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                            print_warning(
                                f"Feature '{col}' has sequence value: {val}, extracting first element")
                            if len(val) > 0:
                                aligned_features[col] = float(val[0])
                            else:
                                aligned_features[col] = 0.0
                        else:
                            # Handle scalar values
                            aligned_features[col] = float(val)
                        feature_matches += 1
                    except Exception as e:
                        print_warning(
                            f"Could not convert feature '{col}' to float: {e}, using 0.0")
                        aligned_features[col] = 0.0

            # Log feature alignment statistics
            print_info(f"Feature alignment: {feature_matches}/{len(feature_cols)} features matched "
                       f"({feature_matches/len(feature_cols)*100:.1f}%)")

            feat_vec = aligned_features
        else:
            # Fallback if no feature columns provided
            print_warning(
                "No feature columns provided. Using all features from DataFrame.")
            feat_vec = last_features

        # Check for missing values
        missing_count = feat_vec.isna().sum()
        if missing_count > 0:
            print_warning(
                f"Found {missing_count} missing values in feature vector. Filling with zeros.")
            feat_vec = feat_vec.fillna(0)

        # Apply scaling if we have a scaler
        if scaler is not None:
            try:
                print_info("Applying feature scaling with StandardScaler")
                # We need to make a copy to avoid a warning about modifying the input
                feat_array = feat_vec.values.copy().reshape(1, -1)
                # Apply transform (not fit_transform!)
                feat_array = scaler.transform(feat_array)
                print_info("Successfully applied feature scaling")

                # Log scaling stats
                if hasattr(scaler, 'mean_'):
                    print_info(
                        f"Scaler features: {len(scaler.mean_)}, Input features: {len(feat_vec)}")

            except Exception as e:
                print_error(f"Error applying scaler: {str(e)}")
                print_warning("Continuing with unscaled features as fallback!")
                feat_array = feat_vec.values.reshape(1, -1)
        else:
            print_warning("No scaler provided! Using unscaled features.")
            # Ensure vec is a numpy array of the right dimension
            feat_array = feat_vec.values
            feat_array = feat_array.reshape(1, -1)

        print_info(
            f"Feature vector shape: {feat_array.shape}, dtype: {feat_array.dtype}")

        # Make prediction with aligned features - handle feature names carefully
        try:
            # Check if feature names match array dimension
            if len(feature_cols) != feat_array.shape[1]:
                print_warning(
                    f"Feature names length ({len(feature_cols)}) doesn't match array width ({feat_array.shape[1]})")
                print_warning(
                    "Creating DMatrix without feature names to avoid dimension mismatch")
                dpredict = xgb.DMatrix(feat_array)
            else:
                print_info(
                    f"Creating DMatrix with feature_names parameter (length={len(feature_cols)})")
                dpredict = xgb.DMatrix(feat_array, feature_names=feature_cols)

            probs = model.predict(dpredict)[0]
            print_info(f"Raw prediction probabilities: {probs}")
        except Exception as e:
            print_error(
                f"Error creating DMatrix or making prediction: {str(e)}")
            # Try fallback approach
            try:
                print_warning(
                    "Trying fallback prediction without feature names")
                dpredict = xgb.DMatrix(feat_array)
                probs = model.predict(dpredict)[0]
                print_info(f"Fallback prediction successful: {probs}")
            except Exception as e2:
                print_error(f"Fallback prediction also failed: {str(e2)}")
                # Return uniform distribution
                num_classes = len(percentiles) + 1
                return {str(i): 1.0 / num_classes for i in range(num_classes)}

        # Format results
        result = {str(i): float(p) for i, p in enumerate(probs)}

    except Exception as e:
        # Handle prediction errors gracefully
        import traceback
        print_error(f"Error during prediction: {str(e)}")
        print_error(f"Error details: {traceback.format_exc()}")

        # Return uniform distribution
        num_classes = len(percentiles) + 1
        result = {str(i): 1.0 / num_classes for i in range(num_classes)}

    # Display a summary of the input streaks
    streak_lengths = [s.get('streak_length', 0) for s in last_streaks]
    streak_means = [s.get('mean_multiplier', 0) for s in last_streaks]

    input_stats = {
        "Number of Streaks": len(last_streaks),
        "Mean Streak Length": f"{np.mean(streak_lengths):.2f}" if streak_lengths else "N/A",
        "Min Streak Length": f"{np.min(streak_lengths):.0f}" if streak_lengths else "N/A",
        "Max Streak Length": f"{np.max(streak_lengths):.0f}" if streak_lengths else "N/A",
        "Mean Multiplier": f"{np.mean(streak_means):.2f}" if streak_means else "N/A",
        "Last Streak Length": f"{streak_lengths[-1]:.0f}" if streak_lengths else "N/A"
    }

    if next_streak_id:
        input_stats["Predicting For"] = f"Streak #{next_streak_id}"

    create_stats_table("Input Streak Summary", input_stats)

    return result
