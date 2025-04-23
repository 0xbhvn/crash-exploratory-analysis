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
                percentiles: List[float] = [0.25, 0.50, 0.75], window: int = 50) -> Tuple[xgb.Booster, Dict[str, float], float]:
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

    Returns:
        Tuple of (trained model, baseline probabilities, empirical hit rate)
    """
    logger.info("Training XGBoost model for streak length prediction")

    # Make a copy to avoid modifying original
    df = df.copy()

    # Time-ordered train/test split preserving streak continuity
    # Use streak_number for ordering
    if 'streak_number' in df.columns:
        df = df.sort_values('streak_number')

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
    percentile_values = [
        df['target_streak_length'].quantile(p) for p in percentiles]

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
        seed=random_seed,
        lookback=window  # Store lookback window size for cross-validation
    )

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
    for fold, (tr_idx, val_idx) in enumerate(rolling_origin_indices(n_train, eval_folds, gap=params.get('lookback', 50)), 1):

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

    # Save the model
    model_path = os.path.join(output_dir, "xgboost_model.pkl")
    model_bundle = {
        "model": bst_final,
        "feature_cols": feature_cols,
        "percentiles": percentiles,
        "percentile_values": [df['target_streak_length'].quantile(p) for p in percentiles],
        "window": window,
        "baseline_probs": baseline_probs,
        "p_hat": p_hat
    }
    joblib.dump(model_bundle, model_path)
    print_success(f"Saved streak-based model to {model_path}")

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
        "Expected Calibration Error": f"{ece_gbm:.4f}",
    }
    create_stats_table("Model Evaluation Metrics", model_metrics)

    # Get full classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Create an improved classification report table with all metrics
    report_table = create_table("Classification Report", [
        "Class", "Precision", "Recall", "F1-Score", "Support"])

    # Add rows for each class
    for class_id in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        class_metrics = report[class_id]
        class_name = f"Class {class_id}"
        add_table_row(report_table, [
            class_name,
            f"{class_metrics['precision']:.4f}",
            f"{class_metrics['recall']:.4f}",
            f"{class_metrics['f1-score']:.4f}",
            f"{class_metrics['support']}"
        ])

    # Add accuracy, macro avg, and weighted avg
    for avg_type in ['accuracy', 'macro avg', 'weighted avg']:
        if avg_type in report:
            metrics = report[avg_type]
            if avg_type == 'accuracy':
                add_table_row(report_table, [
                    avg_type,
                    "",
                    "",
                    f"{metrics:.4f}" if isinstance(
                        metrics, float) else f"{metrics['f1-score']:.4f}",
                    f"{sum(report[c]['support'] for c in report if c not in ['accuracy', 'macro avg', 'weighted avg'])}"
                ])
            else:
                add_table_row(report_table, [
                    avg_type,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    f"{metrics['support']}"
                ])

    # Display the full table
    display_table(report_table)

    # Generate confusion matrix
    generate_confusion_matrix(X_test, y_test, bst_final,
                              feature_cols, output_dir, percentiles)

    # Save test predictions with additional columns
    # Assemble into a dataframe
    test_df_output = test_df.copy()

    # Add probability columns
    for i in range(probs_test.shape[1]):
        test_df_output[f'prob_class{i}'] = probs_test[:, i]

    # Add predicted class as integer
    test_df_output['predicted'] = y_pred.astype(int)

    # Add actual class based on streak_length
    # Add actual class and hit/miss columns
    if 'target_cluster' in test_df_output.columns:
        test_df_output['actual_class'] = test_df_output['target_cluster'].astype(
            int)
    else:
        # Recalculate actual class from streak_length if needed
        conditions = []
        results = []
        for i in range(len(percentiles) + 1):
            if i == 0:
                conditions.append(
                    test_df_output['streak_length'] <= percentile_values[0])
            elif i == len(percentiles):
                conditions.append(
                    test_df_output['streak_length'] > percentile_values[-1])
            else:
                conditions.append(
                    (test_df_output['streak_length'] > percentile_values[i-1]) &
                    (test_df_output['streak_length'] <= percentile_values[i])
                )
            results.append(i)

        test_df_output['actual_class'] = np.select(
            conditions, results, default=np.nan).astype(int)

    # Add hit/miss column (1 if prediction matches actual, 0 otherwise)
    test_df_output['hit_miss'] = (
        test_df_output['predicted'] == test_df_output['actual_class']).astype(int)

    # Remove the descriptive range columns - not needed

    # Save the test predictions
    test_preds_path = os.path.join(output_dir, "test_predictions.csv")
    test_df_output.to_csv(test_preds_path)
    print_info(f"Saved detailed streak test predictions to {test_preds_path}")

    # Calculate and display hit/miss stats
    hit_rate = test_df_output['hit_miss'].mean() * 100
    print_success(f"Overall prediction accuracy: {hit_rate:.2f}%")

    # Print per-class hit rates
    class_hit_rates = test_df_output.groupby(
        'actual_class')['hit_miss'].mean() * 100
    hit_rates_table = create_table(
        "Per-Class Accuracy", ["Class", "Range", "Accuracy"])

    for class_id, hit_rate in class_hit_rates.items():
        class_range = {
            0: f"1-{int(percentile_values[0])}",
            1: f"{int(percentile_values[0])+1}-{int(percentile_values[1])}",
            2: f"{int(percentile_values[1])+1}-{int(percentile_values[2])}",
            3: f">{int(percentile_values[2])}"
        }.get(class_id, "Unknown")

        add_table_row(hit_rates_table, [
            f"Class {int(class_id)}",
            class_range,
            f"{hit_rate:.2f}%"
        ])

    display_table(hit_rates_table)

    return model_bundle


def rolling_origin_indices(n_train: int, n_folds: int, min_val_size: int = 1000, gap: int = 0):
    """
    Generate indices for rolling-origin cross-validation with time series safety.

    Args:
        n_train: Number of training samples
        n_folds: Number of folds
        min_val_size: Minimum size for validation sets
        gap: Gap between training and validation sets (for feature creation)

    Yields:
        Tuple of (train_indices, validation_indices)
    """
    fold_size = n_train // n_folds
    test_size = 5000  # Fixed test size for more consistent evaluation

    for i in range(1, n_folds + 1):
        train_end = min(fold_size * i, n_train - test_size - gap)
        val_start = train_end + gap
        val_end = min(val_start + test_size, n_train)

        # Skip fold if validation set is too small
        if (val_end - val_start) < min_val_size:
            continue

        yield range(0, train_end), range(val_start, val_end)


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
                         percentiles: List[float] = [0.25, 0.50, 0.75]) -> Dict[str, float]:
    """
    Predict the next streak length cluster based on recent streak patterns.

    Args:
        model_or_path: Trained XGBoost model or path to model file
        last_streaks: List of dictionaries with recent streak information
        window: Number of previous streaks to consider
        feature_cols: List of feature column names (if not included in model bundle)
        multiplier_threshold: Threshold for considering a multiplier as a hit (default: 10.0)
        percentiles: List of percentile boundaries for clustering (default: [0.25, 0.50, 0.75])

    Returns:
        Dictionary of cluster probabilities
    """
    from data_processing import make_feature_vector, create_streak_features

    # Load model if path is provided
    model_bundle = None
    if isinstance(model_or_path, str):
        model_bundle = joblib.load(model_or_path)
        model = model_bundle.get("model")
        saved_feature_cols = model_bundle.get("feature_cols")
        if saved_feature_cols:
            feature_cols = saved_feature_cols
    else:
        model = model_or_path  # Already a model object

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

    try:
        # Create DataFrame from streaks
        streak_df = pd.DataFrame(last_streaks)

        # Create features
        features_df = create_streak_features(streak_df, lookback_window=window)

        # Get the last row for prediction
        if features_df.empty:
            raise ValueError("Failed to create features from streak data")

        last_features = features_df.iloc[-1]

        # Align feature vector with the expected feature names from training
        if feature_cols:
            # Create a Series with the expected feature columns, filling missing ones with 0
            aligned_features = pd.Series(0.0, index=feature_cols)

            # Update values where features exist
            for col in feature_cols:
                if col in last_features:
                    aligned_features[col] = last_features[col]

            feat_vec = aligned_features
        else:
            # Fallback if no feature columns provided
            feat_vec = last_features

        # Check for missing values
        if feat_vec.isna().any():
            print_warning(
                "Found missing values in feature vector. Filling with zeros.")
            feat_vec = feat_vec.fillna(0)

        # Make prediction
        dpredict = xgb.DMatrix(feat_vec.values.reshape(
            1, -1), feature_names=feature_cols)
        probs = model.predict(dpredict)[0]

        # Format results
        result = {str(i): float(p) for i, p in enumerate(probs)}

    except Exception as e:
        # Handle prediction errors gracefully
        print_error(f"Error during prediction: {str(e)}")

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
    create_stats_table("Input Streak Summary", input_stats)

    return result
