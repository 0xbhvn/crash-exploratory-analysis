#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model training functionality for temporal analysis.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from logger_config import (
    print_info, print_success, create_table, add_table_row, display_table
)


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

    # Apply data transformations if needed
    X_train_processed, y_train_processed, class_weight_dict = _preprocess_training_data(
        X_train, y_train, use_smote, smote_k_neighbors, use_class_weights, weight_scale
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test)

    # Get sample weights if using class weights
    sample_weights = _get_sample_weights(y_train_processed, class_weight_dict)

    # Create XGBoost data matrices
    dtrain, dtest = _create_xgb_matrices(
        X_train_scaled, y_train_processed, X_test_scaled, y_test, sample_weights)

    # Set model parameters
    params = _get_xgb_params(
        num_classes=len(np.unique(y_train)),
        max_depth=max_depth,
        eta=eta,
        gamma=gamma,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )

    # Display model parameters
    _display_model_params(params)

    # Train model with early stopping
    bst, best_rounds = _train_with_early_stopping(
        params, dtrain, dtest, num_rounds, early_stopping
    )

    # Train final model
    final_model = _train_final_model(params, dtrain, best_rounds)

    # Evaluate model
    y_pred_proba, y_pred, metrics = _evaluate_model(
        final_model, dtest, y_test, y_train)

    # Display evaluation results
    _display_evaluation_results(metrics)
    _display_classification_report(y_test, y_pred)

    # Get feature importance
    importance_df = _get_feature_importance(final_model, feature_cols)

    # Plot feature importance
    feature_importance_plot_path = _plot_feature_importance(
        importance_df, feature_cols, output_dir)

    # Create and save model bundle
    model_bundle = _create_model_bundle(
        final_model, scaler, feature_cols, params, metrics, y_test, y_pred, class_weight_dict,
        use_smote, smote_k_neighbors, X_test, best_rounds
    )

    model_path = _save_model_bundle(model_bundle, output_dir)

    print_success(f"Saved temporal model bundle to {model_path}")
    print_info(
        f"Saved feature importance plot to {feature_importance_plot_path}")

    return final_model, model_bundle


def _preprocess_training_data(X_train, y_train, use_smote, smote_k_neighbors, use_class_weights, weight_scale):
    """
    Preprocess training data with SMOTE or class weights if requested.

    Args:
        X_train: Training features
        y_train: Training targets
        use_smote: Whether to use SMOTE
        smote_k_neighbors: Number of neighbors for SMOTE
        use_class_weights: Whether to use class weights
        weight_scale: Scale factor for class weights

    Returns:
        Processed X_train, y_train, and class_weight_dict (or None)
    """
    class_weight_dict = None

    # Apply SMOTE if requested
    if use_smote:
        print_info(
            f"Applying SMOTE to generate synthetic examples (k_neighbors={smote_k_neighbors})")
        original_shape = X_train.shape
        smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(
            X_train, y_train)

        # Display resampling stats
        _display_smote_results(y_train, y_train_resampled)

        print_info(
            f"Reshaped training data from {original_shape} to {X_train_resampled.shape}")
        return X_train_resampled, y_train_resampled, class_weight_dict

    # Calculate class weights if requested
    if use_class_weights:
        print_info(f"Using class weights with scale factor {weight_scale}")
        class_weight_dict = _calculate_class_weights(y_train, weight_scale)
        _display_class_weights(class_weight_dict, y_train)

    return X_train, y_train, class_weight_dict


def _display_smote_results(y_train, y_train_resampled):
    """
    Display the results of SMOTE resampling.

    Args:
        y_train: Original target values
        y_train_resampled: Resampled target values
    """
    original_counts = pd.Series(y_train).value_counts().sort_index()
    resampled_counts = pd.Series(y_train_resampled).value_counts().sort_index()

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


def _calculate_class_weights(y_train, weight_scale):
    """
    Calculate class weights based on inverse frequency and scale factor.

    Args:
        y_train: Training target values
        weight_scale: Scale factor for weights

    Returns:
        Dictionary mapping class indices to weights
    """
    # Count class frequencies
    class_counts = np.bincount(y_train.astype(np.int64))
    total_samples = len(y_train)
    n_classes = len(class_counts)

    # Calculate inverse frequency weights
    class_weight_dict = {}
    class_freqs = class_counts / total_samples

    for i in range(n_classes):
        # Use inverse frequency weighting adjusted by scale factor
        class_weight = (1 / class_freqs[i]) * weight_scale
        class_weight_dict[i] = class_weight

    return class_weight_dict


def _display_class_weights(class_weight_dict, y_train):
    """
    Display the calculated class weights.

    Args:
        class_weight_dict: Dictionary mapping class indices to weights
        y_train: Training target values
    """
    class_counts = np.bincount(y_train.astype(np.int64))
    total_samples = len(y_train)
    class_freqs = class_counts / total_samples

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


def _get_sample_weights(y_train, class_weight_dict):
    """
    Get sample weights for training instances based on class weights.

    Args:
        y_train: Training target values
        class_weight_dict: Dictionary mapping class indices to weights

    Returns:
        Sample weights array or None
    """
    if class_weight_dict is None:
        return None

    # Create sample weights for each training instance
    return np.array([class_weight_dict[y] for y in y_train])


def _create_xgb_matrices(X_train_scaled, y_train, X_test_scaled, y_test, sample_weights):
    """
    Create XGBoost DMatrix objects for training and testing.

    Args:
        X_train_scaled: Scaled training features
        y_train: Training targets
        X_test_scaled: Scaled testing features
        y_test: Testing targets
        sample_weights: Sample weights for training instances or None

    Returns:
        Training and testing DMatrix objects
    """
    dtrain = xgb.DMatrix(
        X_train_scaled,
        label=y_train,
        weight=sample_weights,
        feature_names=[f'f{i}' for i in range(X_train_scaled.shape[1])]
    )

    dtest = xgb.DMatrix(
        X_test_scaled,
        label=y_test,
        feature_names=[f'f{i}' for i in range(X_test_scaled.shape[1])]
    )

    return dtrain, dtest


def _get_xgb_params(num_classes, max_depth, eta, gamma, min_child_weight,
                    reg_lambda, subsample, colsample_bytree):
    """
    Get XGBoost parameters.

    Args:
        num_classes: Number of target classes
        max_depth: Maximum tree depth
        eta: Learning rate
        gamma: Minimum loss reduction
        min_child_weight: Minimum sum of instance weight in child
        reg_lambda: L2 regularization
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns per tree

    Returns:
        Dictionary of XGBoost parameters
    """
    return {
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


def _display_model_params(params):
    """
    Display model parameters in a table.

    Args:
        params: Dictionary of model parameters
    """
    param_table = create_table("XGBoost Parameters", ["Parameter", "Value"])
    for param, value in params.items():
        add_table_row(param_table, [param, str(value)])
    display_table(param_table)


def _train_with_early_stopping(params, dtrain, dtest, num_rounds, early_stopping):
    """
    Train the model with early stopping.

    Args:
        params: XGBoost parameters
        dtrain: Training DMatrix
        dtest: Testing DMatrix
        num_rounds: Maximum number of boosting rounds
        early_stopping: Early stopping rounds

    Returns:
        Trained model and best number of rounds
    """
    print_info("Training model with validation-based early stopping")

    evallist = [(dtrain, 'training'), (dtest, 'validation')]

    bst = xgb.train(
        params,
        dtrain,
        num_rounds,
        evals=evallist,
        early_stopping_rounds=early_stopping,
        verbose_eval=100
    )

    best_rounds = bst.best_iteration
    print_info(f"Best number of rounds: {best_rounds}")

    return bst, best_rounds


def _train_final_model(params, dtrain, best_rounds):
    """
    Train the final model with the best number of rounds.

    Args:
        params: XGBoost parameters
        dtrain: Training DMatrix
        best_rounds: Best number of rounds

    Returns:
        Final trained model
    """
    print_info("Training final model on full training data")
    final_model = xgb.train(
        params,
        dtrain,
        best_rounds
    )

    return final_model


def _evaluate_model(model, dtest, y_test, y_train):
    """
    Evaluate the model and calculate performance metrics.

    Args:
        model: Trained XGBoost model
        dtest: Testing DMatrix
        y_test: True test labels
        y_train: True train labels

    Returns:
        Prediction probabilities, predicted labels, and metrics dictionary
    """
    # Make predictions
    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Calculate baseline log loss (prediction based on class frequencies)
    y_train_int = y_train.astype(np.int64)
    class_weights = np.bincount(y_train_int) / len(y_train)
    baseline_probs = np.tile(class_weights, (len(y_test), 1))
    baseline_logloss = log_loss(y_test, baseline_probs)
    logloss_improvement = (baseline_logloss - logloss) / baseline_logloss * 100

    metrics = {
        'accuracy': accuracy,
        'log_loss': logloss,
        'baseline_log_loss': baseline_logloss,
        'log_loss_improvement': logloss_improvement
    }

    return y_pred_proba, y_pred, metrics


def _display_evaluation_results(metrics):
    """
    Display evaluation results in a table.

    Args:
        metrics: Dictionary of evaluation metrics
    """
    eval_table = create_table("Temporal Model Evaluation", ["Metric", "Value"])
    add_table_row(eval_table, ["Accuracy", f"{metrics['accuracy']:.4f}"])
    add_table_row(eval_table, ["Log Loss", f"{metrics['log_loss']:.4f}"])
    add_table_row(eval_table, ["Baseline Log Loss",
                  f"{metrics['baseline_log_loss']:.4f}"])
    add_table_row(eval_table, ["Log Loss Improvement",
                  f"{metrics['log_loss_improvement']:.2f}%"])
    display_table(eval_table)


def _display_classification_report(y_test, y_pred):
    """
    Display classification report in a table.

    Args:
        y_test: True test labels
        y_pred: Predicted labels
    """
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


def _get_feature_importance(model, feature_cols):
    """
    Get feature importance from the model.

    Args:
        model: Trained XGBoost model
        feature_cols: Feature column names

    Returns:
        DataFrame with feature importance information
    """
    # Get feature importance
    importances = model.get_score(importance_type='gain')
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

    return importance_df


def _plot_feature_importance(importance_df, feature_cols, output_dir):
    """
    Plot feature importance and save to file.

    Args:
        importance_df: DataFrame with feature importance
        feature_cols: Feature column names
        output_dir: Directory to save the plot

    Returns:
        Path to the saved plot
    """
    top_n = min(10, len(importance_df))
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(12, 8))
    importance_with_names = top_features.copy()
    importance_with_names['DisplayName'] = importance_with_names['Feature'].apply(
        lambda f: f"{f} ({feature_cols[int(f.replace('f', ''))]})" if f.startswith(
            'f') else f
    )
    plt.barh(importance_with_names['DisplayName'],
             importance_with_names['Importance'])
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top

    # Save feature importance plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'temporal_feature_importance.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return plot_path


def _create_model_bundle(model, scaler, feature_cols, params, metrics, y_test,
                         y_pred, class_weight_dict, use_smote, smote_k_neighbors, X_test, best_rounds):
    """
    Create a model bundle with all metadata for saving.

    Args:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
        feature_cols: Feature column names
        params: XGBoost parameters
        metrics: Evaluation metrics
        y_test: True test labels
        y_pred: Predicted test labels
        class_weight_dict: Class weight dictionary or None
        use_smote: Whether SMOTE was used
        smote_k_neighbors: Number of neighbors used for SMOTE
        X_test: Test features
        best_rounds: Best number of rounds

    Returns:
        Model bundle dictionary
    """
    return {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'params': params,
        'num_classes': len(np.unique(y_test)),
        'feature_importance': model.get_score(importance_type='gain'),
        'metrics': metrics,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_indices': X_test.index.tolist(),
        'test_labels': y_test.tolist(),
        'predictions': y_pred.tolist(),
        'class_weights': class_weight_dict,
        'used_smote': use_smote,
        'smote_params': {'k_neighbors': smote_k_neighbors} if use_smote else None,
        'best_rounds': best_rounds
    }


def _save_model_bundle(model_bundle, output_dir):
    """
    Save the model bundle to disk.

    Args:
        model_bundle: Model bundle dictionary
        output_dir: Directory to save the model

    Returns:
        Path to the saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'temporal_model.pkl')
    joblib.dump(model_bundle, model_path)

    return model_path
