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
                percentiles: List[float] = [0.25, 0.50, 0.75]) -> Tuple[xgb.Booster, Dict[str, float], float]:
    """
    Train an XGBoost model for streak length prediction.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        test_frac: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        eval_folds: Number of folds for cross-validation
        output_dir: Directory to save model and outputs
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])

    Returns:
        Tuple of (trained model, baseline probabilities, empirical hit rate)
    """
    logger.info("Training XGBoost model for streak length prediction")

    # Train/test split
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Show split information
    split_info = {
        "Training Samples": len(train_df),
        "Testing Samples": len(test_df),
        "Training Split": f"{(1-test_frac)*100:.1f}%",
        "Testing Split": f"{test_frac*100:.1f}%"
    }
    create_stats_table("Train/Test Split", split_info)

    X_train, y_train = train_df[feature_cols], train_df["target_cluster"]
    X_test, y_test = test_df[feature_cols], test_df["target_cluster"]

    # Calculate baseline probabilities
    unique_classes = sorted(y_train.unique())
    baseline_probs = {}
    for c in unique_classes:
        baseline_probs[c] = (y_train == c).mean()

    print_info(
        f"Baseline probabilities: {', '.join([f'Class {k}: {v:.3f}' for k, v in baseline_probs.items()])}")

    # Calculate empirical hit rate
    hit_col = "is_hit{}".format(int(
        multiplier_threshold) if multiplier_threshold.is_integer() else multiplier_threshold)
    if hit_col in df.columns:
        p_hat = df[hit_col].mean()
    else:
        # Fallback if column doesn't exist
        p_hat = (df['Bust'] >= multiplier_threshold).mean()
    print_info(f"Empirical {multiplier_threshold}× hit rate: {p_hat:.4f}")

    # Display baseline probabilities
    baseline_table = create_table("Baseline Probabilities", [
                                  "Cluster", "Description", "Probability"])

    # Create dynamic cluster descriptions based on percentiles
    cluster_descriptions = {}
    # Calculate the actual streak length values at each percentile
    percentile_values = [
        df['next_streak_length'].quantile(p) for p in percentiles]

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

    # Perform rolling-origin cross-validation
    n_train = len(X_train)
    dtrain_full = xgb.DMatrix(X_train, label=y_train,
                              feature_names=feature_cols)

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

    # Display model parameters
    param_table = create_table("XGBoost Parameters", ["Parameter", "Value"])
    for param, value in params.items():
        add_table_row(param_table, [param, str(value)])
    display_table(param_table)

    print_info(f"Performing {eval_folds}-fold rolling-origin cross-validation")

    # Create a table for CV results
    cv_table = create_table("Cross-Validation Results",
                            ["Fold", "Best Round", "Log Loss"])

    best_ntrees = []
    for fold, (tr_idx, val_idx) in enumerate(
            rolling_origin_indices(n_train, eval_folds), 1):

        dtr = xgb.DMatrix(X_train.iloc[tr_idx], label=y_train.iloc[tr_idx],
                          feature_names=feature_cols)
        dval = xgb.DMatrix(X_train.iloc[val_idx], label=y_train.iloc[val_idx],
                           feature_names=feature_cols)

        bst = xgb.train(
            params, dtr, num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )

        br = best_round(bst, default_rounds=2000)
        best_ntrees.append(br)

        # Add row to CV table
        best_score = bst.best_score if hasattr(bst, 'best_score') else "N/A"
        add_table_row(cv_table, [fold, br, f"{best_score:.4f}" if isinstance(
            best_score, float) else best_score])

    display_table(cv_table)

    nrounds = int(np.median(best_ntrees))
    print_info(f"Using n_rounds = {nrounds} (median from cross-validation)")

    # Final fit on all training data
    print_info("Training final model on all training data")
    bst_final = xgb.train(params, dtrain_full, num_boost_round=nrounds)

    # Save the model
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    joblib.dump(bst_final, model_path)
    print_success(f"Saved model to {model_path}")

    # Test-set evaluation
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    probs_test = bst_final.predict(dtest)
    y_pred = np.argmax(probs_test, axis=1)

    # Calculate metrics
    logloss_gbm = log_loss(y_test, probs_test)
    ece_gbm = expected_calibration_error(y_test, probs_test)

    # Display evaluation metrics
    uplift = (logloss_baseline - logloss_gbm) / logloss_baseline * 100
    model_metrics = {
        "Log Loss (Baseline)": f"{logloss_baseline:.4f}",
        "Log Loss (Model)": f"{logloss_gbm:.4f}",
        "Log Loss Improvement": f"{uplift:.2f}%",
        "Expected Calibration Error": f"{ece_gbm:.4f}"
    }
    create_stats_table("Model Evaluation Metrics", model_metrics)

    # Display classification report as a rich table
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create classification report table
    class_table = create_table("Classification Report",
                               ["Class", "Precision", "Recall", "F1-Score", "Support"])

    # Add rows for each class
    for class_id in sorted([k for k in report.keys() if k.isdigit()]):
        metrics = report[class_id]
        add_table_row(class_table, [
            class_id,
            f"{metrics['precision']:.2f}",
            f"{metrics['recall']:.2f}",
            f"{metrics['f1-score']:.2f}",
            f"{metrics['support']}"
        ])

    # Add accuracy row
    if 'accuracy' in report:
        add_table_row(class_table, [
                      "accuracy", "", "", f"{report['accuracy']:.2f}", f"{sum(report[c]['support'] for c in report if c.isdigit())}"])

    display_table(class_table)

    # Generate confusion matrix
    generate_confusion_matrix(
        X_test, y_test, bst_final, feature_cols, output_dir, percentiles)

    return bst_final, baseline_probs, p_hat


def rolling_origin_indices(n_train: int, n_folds: int):
    """
    Generate indices for rolling-origin cross-validation.

    Args:
        n_train: Number of training samples
        n_folds: Number of folds

    Yields:
        Tuple of (train_indices, validation_indices)
    """
    fold_size = n_train // n_folds
    for i in range(1, n_folds + 1):
        split = fold_size * i
        yield range(0, split), range(split, min(split + fold_size, n_train))


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
        if 'next_streak_length' in train_df.columns:
            percentile_values = [
                train_df['next_streak_length'].quantile(p) for p in percentiles]
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


def predict_next_cluster(model, last_window_multipliers: List[float], window: int,
                         feature_cols: List[str], multiplier_threshold: float = 10.0,
                         percentiles: List[float] = [0.25, 0.50, 0.75]) -> Dict[str, float]:
    """
    Predict the next cluster based on recent multipliers.

    Args:
        model: Trained XGBoost model
        last_window_multipliers: List of last WINDOW multipliers
        window: Rolling window size
        feature_cols: List of feature column names
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])

    Returns:
        Dictionary of cluster probabilities
    """
    from data_processing import make_feature_vector

    # Check input length
    if len(last_window_multipliers) != window:
        print_warning(
            f"Expected {window} multipliers, but got {len(last_window_multipliers)}. Results may be inaccurate.")

    # Create feature vector
    feat_vec = make_feature_vector(
        last_window_multipliers, window, feature_cols)

    # Make prediction
    dpredict = xgb.DMatrix(feat_vec.values.reshape(
        1, -1), feature_names=feature_cols)
    probs = model.predict(dpredict)[0]

    # Format results
    result = {str(i): float(p) for i, p in enumerate(probs)}

    # Display a summary of the input multipliers
    input_stats = {
        "Input Length": len(last_window_multipliers),
        "Mean Multiplier": f"{np.mean(last_window_multipliers):.2f}",
        "Min Multiplier": f"{np.min(last_window_multipliers):.2f}",
        "Max Multiplier": f"{np.max(last_window_multipliers):.2f}",
        f"{multiplier_threshold}× Count": sum(1 for x in last_window_multipliers if x >= multiplier_threshold),
        f"{multiplier_threshold}× Rate": f"{sum(1 for x in last_window_multipliers if x >= multiplier_threshold) / len(last_window_multipliers) * 100:.2f}%"
    }
    create_stats_table("Input Multiplier Summary", input_stats)

    return result
