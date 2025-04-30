#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction functionality for temporal analysis.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Optional
from utils.logger_config import print_info
from temporal.features import create_temporal_features


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
    # Extract model components
    model, scaler, feature_cols = _extract_model_components(model_bundle)

    # Prepare data with temporal indices
    features_df = _prepare_streak_data(recent_streaks, temporal_idx_start)

    # Extract and scale features
    X_scaled = _extract_and_scale_features(
        features_df, feature_cols, scaler, model_bundle)

    # Make predictions
    y_pred, y_pred_proba = _make_model_predictions(model, X_scaled)

    # Add predictions to results
    features_df = _add_predictions_to_results(
        features_df, y_pred, y_pred_proba, model_bundle['num_classes']
    )

    # Add streak information
    features_df = _add_streak_information(features_df, recent_streaks)

    return features_df


def _extract_model_components(model_bundle):
    """
    Extract necessary components from the model bundle.

    Args:
        model_bundle: Dictionary containing model and preprocessing info

    Returns:
        Tuple of (model, scaler, feature_cols)
    """
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']

    return model, scaler, feature_cols


def _prepare_streak_data(recent_streaks, temporal_idx_start):
    """
    Prepare streak data with temporal indices and features.

    Args:
        recent_streaks: DataFrame with recent streak data
        temporal_idx_start: Starting temporal index for the new data

    Returns:
        DataFrame with temporal features
    """
    # Ensure we have a temporal index
    if temporal_idx_start is None:
        temporal_idx_start = 0

    # Assign sequential temporal indices
    recent_streaks_copy = recent_streaks.copy()
    recent_streaks_copy['temporal_idx'] = range(
        temporal_idx_start,
        temporal_idx_start + len(recent_streaks_copy)
    )

    # Create temporal features
    features_df, _, _ = create_temporal_features(
        recent_streaks_copy, lookback_window=5)

    return features_df


def _extract_and_scale_features(features_df, feature_cols, scaler, model_bundle):
    """
    Extract and scale features for prediction.

    Args:
        features_df: DataFrame with features
        feature_cols: List of feature column names
        scaler: Fitted StandardScaler
        model_bundle: Model bundle with feature information

    Returns:
        Scaled feature matrix as Numpy array
    """
    # Check for missing columns and add them with default values
    missing_cols = [
        col for col in feature_cols if col not in features_df.columns]

    if missing_cols:
        print_info(
            f"Missing {len(missing_cols)} feature columns. Adding with default values.")
        for col in missing_cols:
            features_df[col] = 0

    # Extract feature values and scale
    X = features_df[feature_cols]
    X_scaled = scaler.transform(X)

    # Create DMatrix for prediction - REMOVED
    # dpredict = xgb.DMatrix(
    #     X_scaled,
    #     feature_names=[f'f{i}' for i in range(X_scaled.shape[1])]
    # )

    # Return the scaled numpy array directly
    return X_scaled


def _make_model_predictions(model, X_scaled):
    """
    Make predictions using the model.

    Args:
        model: Trained model (CalibratedClassifierCV)
        X_scaled: Scaled features as a NumPy array

    Returns:
        Tuple of (predicted classes, prediction probabilities)
    """
    # Make predictions
    # Use predict_proba as the calibrator works on probabilities
    y_pred_proba = model.predict_proba(X_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred, y_pred_proba


def _add_predictions_to_results(features_df, y_pred, y_pred_proba, num_classes):
    """
    Add prediction results to the features DataFrame.

    Args:
        features_df: DataFrame with features
        y_pred: Predicted classes
        y_pred_proba: Prediction probabilities
        num_classes: Number of possible classes

    Returns:
        DataFrame with prediction results added
    """
    # Add predicted cluster
    features_df['predicted_cluster'] = y_pred

    # Add probability for each class
    for i in range(num_classes):
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
    class_probs = [features_df[f'prob_class_{i}'] for i in range(num_classes)]
    features_df['prediction_confidence'] = np.max(
        np.column_stack(class_probs), axis=1)

    return features_df


def _add_streak_information(features_df, recent_streaks):
    """
    Add streak information from the original streaks DataFrame.

    Args:
        features_df: DataFrame with predictions
        recent_streaks: Original streaks DataFrame

    Returns:
        DataFrame with streak information added
    """
    # Add game IDs and streak information
    features_df['start_game_id'] = recent_streaks['start_game_id']
    features_df['end_game_id'] = recent_streaks['end_game_id']
    features_df['streak_number'] = recent_streaks.index
    features_df['streak_length'] = recent_streaks['streak_length']

    return features_df
