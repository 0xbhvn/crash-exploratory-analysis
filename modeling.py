#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modeling module for Crash Game 10× Streak Analysis.

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

# Import rich logging
from logger_config import (
    console, create_table, display_table, add_table_row,
    create_stats_table, print_info, print_success, print_warning,
    print_error, print_panel
)

logger = logging.getLogger(__name__)


def train_model(df: pd.DataFrame, feature_cols: List[str], clusters: Dict[int, Tuple[int, int]],
                test_frac: float, random_seed: int, eval_folds: int = 5,
                output_dir: str = './output') -> Tuple[xgb.Booster, Dict[str, float], float]:
    """
    Train a model to predict streak length clusters.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        clusters: Dictionary mapping cluster IDs to (min, max) streak length ranges
        test_frac: Fraction of data to use for testing
        random_seed: Random seed for reproducibility
        eval_folds: Number of folds for rolling-origin cross-validation
        output_dir: Directory to save outputs

    Returns:
        Tuple of (trained XGBoost model, baseline probabilities, p_hat value)
    """
    print_info("Training model to predict streak length clusters")

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

    # Baseline geometric
    p_hat = (df["Bust"] >= 10).mean()
    baseline_probs = {
        c: (1 - p_hat)**(lo-1) - (1 - p_hat)**hi
        for c, (lo, hi) in clusters.items()
    }

    # Display baseline probabilities
    baseline_table = create_table("Baseline Probabilities", [
                                  "Cluster", "Description", "Probability"])
    for cluster_id, prob in baseline_probs.items():
        cluster_range = clusters[cluster_id]
        cluster_desc = f"Streak Length {cluster_range[0]}-{cluster_range[1]}"
        if cluster_range[1] > 1000:
            cluster_desc = f"Streak Length {cluster_range[0]}+"
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

    params = dict(
        objective="multi:softprob",
        num_class=len(clusters),
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
        X_test, y_test, bst_final, feature_cols, output_dir)

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


def generate_confusion_matrix(X_test, y_test, model, feature_cols, output_dir) -> None:
    """
    Generate and save confusion matrix for test set.

    Args:
        X_test: Test features
        y_test: Test labels
        model: Trained XGBoost model
        feature_cols: List of feature column names
        output_dir: Directory to save outputs
    """
    # Prepare data
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    probs = model.predict(dtest)

    # Assemble into a DataFrame
    results = pd.DataFrame(probs,
                           columns=["prob_short", "prob_medium", "prob_long"],
                           index=X_test.index)
    results["actual"] = y_test.values
    results["predicted"] = results[["prob_short", "prob_medium", "prob_long"]].idxmax(axis=1) \
        .map({"prob_short": 0, "prob_medium": 1, "prob_long": 2})

    # Generate confusion matrix
    cm = confusion_matrix(results["actual"], results["predicted"])

    # Display confusion matrix as a table
    cm_table = create_table("Confusion Matrix", [
                            ""] + [f"Predicted {i}" for i in range(cm.shape[1])])

    # Add rows for the confusion matrix
    for i in range(cm.shape[0]):
        row_data = [f"Actual {i}"] + [str(cm[i, j])
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
            cm_metrics, [str(i), f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])

    display_table(cm_metrics)

    # Save detailed results to CSV
    results_path = os.path.join(output_dir, "test_predictions.csv")
    results.to_csv(results_path)
    print_info(f"Saved detailed test predictions to {results_path}")


def predict_next_cluster(model, last_window_multipliers: List[float], window: int,
                         feature_cols: List[str]) -> Dict[str, float]:
    """
    Predict the next cluster based on recent multipliers.

    Args:
        model: Trained XGBoost model
        last_window_multipliers: List of last WINDOW multipliers
        window: Rolling window size
        feature_cols: List of feature column names

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
        "10× Count": sum(1 for x in last_window_multipliers if x >= 10),
        "10× Rate": f"{sum(1 for x in last_window_multipliers if x >= 10) / len(last_window_multipliers) * 100:.2f}%"
    }
    create_stats_table("Input Multiplier Summary", input_stats)

    return result
