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
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional, Any

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
    logger.info("Training model to predict streak length clusters")

    # Train/test split
    split_idx = int(len(df) * (1 - test_frac))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols], train_df["target_cluster"]
    X_test, y_test = test_df[feature_cols], test_df["target_cluster"]

    # Baseline geometric
    p_hat = (df["Bust"] >= 10).mean()
    baseline_probs = {
        c: (1 - p_hat)**(lo-1) - (1 - p_hat)**hi
        for c, (lo, hi) in clusters.items()
    }

    baseline_pred = np.tile(list(baseline_probs.values()), (len(y_test), 1))
    logloss_baseline = log_loss(y_test, baseline_pred)
    logger.info(f"Baseline geometric log-loss: {logloss_baseline:.4f}")

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
        logger.info(f"Fold {fold}: best_nrounds = {br}")

    nrounds = int(np.median(best_ntrees))
    logger.info(f"Using n_rounds = {nrounds}")

    # Final fit on all training data
    bst_final = xgb.train(params, dtrain_full, num_boost_round=nrounds)

    # Save the model
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    joblib.dump(bst_final, model_path)
    logger.info(f"Saved model to {model_path}")

    # Test-set evaluation
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)
    probs_test = bst_final.predict(dtest)
    logloss_gbm = log_loss(y_test, probs_test)

    ece_gbm = expected_calibration_error(y_test, probs_test)

    uplift = (logloss_baseline - logloss_gbm) / logloss_baseline * 100
    logger.info(f"GBM log-loss: {logloss_gbm:.4f} | ECE: {ece_gbm:.4f}")
    logger.info(f"Log-loss reduction vs baseline: {uplift:.2f}%")

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

    # Confusion matrix (row-normalized)
    conf_mat = pd.crosstab(results["predicted"], results["actual"],
                           rownames=["Predicted"], colnames=["Actual"],
                           normalize="index").round(3)

    # Add sample counts per predicted bucket
    conf_mat["n_samples"] = pd.crosstab(
        results["predicted"], results["actual"]).sum(axis=1)

    # Save confusion matrix
    conf_mat.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    logger.info("Saved confusion matrix to confusion_matrix.csv")


def predict_next_cluster(model, last_window_multipliers: List[float], window: int,
                         feature_cols: List[str]) -> Dict[str, float]:
    """
    Predict the next cluster based on recent multipliers.

    Args:
        model: Trained XGBoost model
        last_window_multipliers: List of last window multipliers
        window: Window size used for feature engineering
        feature_cols: List of feature column names

    Returns:
        Dictionary of cluster probabilities
    """
    from data_processing import make_feature_vector

    if len(last_window_multipliers) != window:
        raise ValueError(
            f"Expected {window} multipliers, got {len(last_window_multipliers)}")

    vec = make_feature_vector(last_window_multipliers, window, feature_cols)
    d = xgb.DMatrix(vec.values.reshape(1, -1), feature_names=feature_cols)
    probs = model.predict(d)[0]

    return {
        "short_≤5": float(probs[0]),
        "medium_6‑12": float(probs[1]),
        "long_>12": float(probs[2]),
    }
