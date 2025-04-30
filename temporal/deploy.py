#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deployment functionality for temporal streak prediction model.

This module provides functions to deploy the temporal model for predicting
the next streak based on the most recent streak data, ensuring strict
temporal boundaries with no data leakage.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from typing import Dict, List, Tuple, Any, Optional
from utils.logger_config import (
    print_info, print_success, print_warning, print_error,
    create_table, add_table_row, display_table
)


def predict_next_streak(model_bundle: Dict, latest_streaks: pd.DataFrame) -> Dict:
    """
    Predict the next streak after the latest available data.

    Args:
        model_bundle: Trained model bundle with model, scaler and feature info
        latest_streaks: DataFrame with recent streak data

    Returns:
        Dictionary with prediction details for the next streak
    """
    # Extract model components
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']

    # Use only historical data for feature creation
    historical_streaks = latest_streaks.copy()

    # Create a "dummy" next streak as a placeholder
    next_streak = pd.DataFrame({
        'streak_number': [historical_streaks['streak_number'].max() + 1],
        'start_game_id': [historical_streaks['end_game_id'].max() + 1],
        # Will be updated later
        'end_game_id': [historical_streaks['end_game_id'].max() + 1],
        'streak_length': [0],  # Will be predicted
        'temporal_idx': [historical_streaks['temporal_idx'].max() + 1 if 'temporal_idx' in historical_streaks.columns else len(historical_streaks)]
    })

    # Display information about the prediction
    print_info(
        f"Predicting next streak after game ID {historical_streaks['end_game_id'].max()}")

    # Create strictly temporal features using only historical data
    features = _create_strictly_temporal_features(
        historical_streaks,
        next_streak,
        lookback_window=5
    )

    # Handle missing feature columns
    missing_cols = [col for col in feature_cols if col not in features.columns]
    if missing_cols:
        print_warning(
            f"Missing {len(missing_cols)} feature columns. Adding with default values.")
        for col in missing_cols:
            features[col] = 0

    # Extract and scale features
    X = features[feature_cols]
    X_scaled = scaler.transform(X)

    # Make prediction using the loaded model (Calibrator)
    print_info("Using calibrated model for prediction probabilities.")
    y_pred_proba = model.predict_proba(X_scaled)

    y_pred = np.argmax(y_pred_proba, axis=1)
    confidence = np.max(y_pred_proba, axis=1)[0]

    # Map prediction to descriptive category
    cluster_to_desc = {
        0: 'short (1-3)',
        1: 'medium_short (4-7)',
        2: 'medium_long (8-14)',
        3: 'long (>14)'
    }

    length_ranges = {
        0: (1, 3),
        1: (4, 7),
        2: (8, 14),
        3: (15, float('inf'))
    }

    predicted_cluster = int(y_pred[0])

    # Create prediction result
    prediction = {
        'next_streak_number': int(next_streak['streak_number'][0]),
        'starts_after_game_id': int(next_streak['start_game_id'][0] - 1),
        'predicted_cluster': predicted_cluster,
        'prediction_desc': cluster_to_desc[predicted_cluster],
        'predicted_length_range': length_ranges[predicted_cluster],
        'confidence': float(confidence)
    }

    # Add probabilities for each class
    for i in range(len(y_pred_proba[0])):
        prediction[f'prob_class_{i}'] = float(y_pred_proba[0][i])

    # Display the prediction
    _display_prediction_result(prediction, historical_streaks)

    return prediction


def _create_strictly_temporal_features(
    historical_streaks: pd.DataFrame,
    next_streak: pd.DataFrame,
    lookback_window: int = 5
) -> pd.DataFrame:
    """
    Create strictly temporal features for the next streak using only historical data.

    Args:
        historical_streaks: DataFrame with historical streak data
        next_streak: DataFrame with the next streak to predict
        lookback_window: Number of previous streaks to use for features

    Returns:
        DataFrame with temporal features for the next streak
    """
    # Initialize features DataFrame with the next streak
    features = next_streak.copy()

    # Get the most recent historical streaks up to lookback_window
    recent_history = historical_streaks.tail(lookback_window).copy()

    # Create lagged features
    for i in range(1, min(lookback_window, len(recent_history)) + 1):
        if i <= len(recent_history):
            historical_idx = len(recent_history) - i
            features[f'prev{i}_length'] = recent_history.iloc[historical_idx]['streak_length']
            if 'hit_multiplier' in recent_history.columns:
                features[f'prev{i}_hit_mult'] = recent_history.iloc[historical_idx]['hit_multiplier']
            else:
                # Default to 10.0 if hit_multiplier is missing
                features[f'prev{i}_hit_mult'] = 10.0

            # Create streak length difference features
            if i > 1:
                prev_length = features[f'prev{i-1}_length'].iloc[0]
                current_length = features[f'prev{i}_length'].iloc[0]
                features[f'diff{i-1}_to_{i}'] = prev_length - current_length

    # Calculate rolling statistics
    lengths = []
    hit_mults = []

    # Collect past streak lengths and hit multipliers in reverse order (most recent first)
    for i in range(min(lookback_window, len(recent_history))):
        idx = len(recent_history) - 1 - i
        lengths.append(recent_history.iloc[idx]['streak_length'])
        if 'hit_multiplier' in recent_history.columns:
            hit_mults.append(recent_history.iloc[idx]['hit_multiplier'])
        else:
            hit_mults.append(10.0)  # Default value if missing

    # Calculate rolling statistics for different window sizes
    for window in [3, 5]:
        if window <= len(lengths):
            # Rolling mean
            features[f'rolling_mean_{window}'] = np.mean(lengths[:window])

            # Rolling standard deviation
            if len(lengths[:window]) > 1:
                features[f'rolling_std_{window}'] = np.std(lengths[:window])
            else:
                features[f'rolling_std_{window}'] = 0

            # Rolling max and min
            features[f'rolling_max_{window}'] = np.max(lengths[:window])
            features[f'rolling_min_{window}'] = np.min(lengths[:window])

            # Category percentages
            short_count = sum(1 for l in lengths[:window] if l <= 3)
            medium_count = sum(1 for l in lengths[:window] if 3 < l <= 14)
            long_count = sum(1 for l in lengths[:window] if l > 14)

            features[f'short_pct_{window}'] = short_count / window
            features[f'medium_pct_{window}'] = medium_count / window
            features[f'long_pct_{window}'] = long_count / window

            # Hit multiplier rolling stats
            if len(hit_mults[:window]) > 0:
                features[f'hit_mult_mean_{window}'] = np.mean(
                    hit_mults[:window])
                if len(hit_mults) > window:
                    features[f'hit_mult_trend_{window}'] = features[f'hit_mult_mean_{window}'].iloc[0] - hit_mults[window]
                else:
                    features[f'hit_mult_trend_{window}'] = 0

    # Add category features
    categories = []
    for length in lengths:
        if length <= 3:
            categories.append('short')
        elif length <= 7:
            categories.append('medium_short')
        elif length <= 14:
            categories.append('medium_long')
        else:
            categories.append('long')

    # Time since features
    time_since = {
        'short': 99,
        'medium_short': 99,
        'medium_long': 99,
        'long': 99
    }

    # Calculate time since each category
    for i, cat in enumerate(categories):
        time_since[cat] = i

    for cat in time_since:
        features[f'time_since_{cat}'] = time_since[cat]

    # Category run length features
    if len(categories) > 0:
        run_counter = 1
        prev_cat = categories[0]

        for i in range(1, len(categories)):
            if categories[i] == prev_cat:
                run_counter += 1
            else:
                run_counter = 1
                prev_cat = categories[i]

        features['category_run_length'] = run_counter
        features['prev_run_length'] = run_counter
    else:
        features['category_run_length'] = 1
        features['prev_run_length'] = 1

    # One-hot category features
    for cat in ['short', 'medium_short', 'medium_long', 'long']:
        if len(categories) > 0:
            features[f'prev_cat_{cat}'] = 1 if categories[0] == cat else 0
        else:
            features[f'prev_cat_{cat}'] = 0

    # Set 'same_as_prev' feature (default to 0 as we don't know the next category yet)
    features['same_as_prev'] = 0

    return features


def _display_prediction_result(prediction: Dict, historical_streaks: pd.DataFrame) -> None:
    """
    Display the prediction result in a formatted table.

    Args:
        prediction: Dictionary with prediction details
        historical_streaks: DataFrame with historical streak data
    """
    # Create a prediction overview table
    pred_table = create_table(
        "Next Streak Prediction",
        ["Metric", "Value"]
    )

    add_table_row(pred_table, [
        "Next Streak Number",
        f"{prediction['next_streak_number']}"
    ])

    add_table_row(pred_table, [
        "Starts After Game ID",
        f"{prediction['starts_after_game_id']}"
    ])

    add_table_row(pred_table, [
        "Predicted Length Range",
        f"{prediction['prediction_desc']}"
    ])

    add_table_row(pred_table, [
        "Confidence",
        f"{prediction['confidence']:.4f}"
    ])

    display_table(pred_table)

    # Create a table of class probabilities
    prob_table = create_table(
        "Class Probabilities",
        ["Class", "Description", "Probability"]
    )

    cluster_to_desc = {
        0: 'Short (1-3)',
        1: 'Medium-Short (4-7)',
        2: 'Medium-Long (8-14)',
        3: 'Long (>14)'
    }

    for i in range(4):  # Assuming 4 classes
        add_table_row(prob_table, [
            f"Class {i}",
            cluster_to_desc[i],
            f"{prediction[f'prob_class_{i}']:.4f}"
        ])

    display_table(prob_table)

    # Show recent history used for prediction
    history_table = create_table(
        "Recent Streak History (Used for Prediction)",
        ["Streak #", "Game Range", "Length", "Category"]
    )

    recent_history = historical_streaks.tail(5)  # Last 5 streaks

    for _, row in recent_history.iterrows():
        streak_length = row['streak_length']
        category = "Short (1-3)"
        if streak_length > 3 and streak_length <= 7:
            category = "Medium-Short (4-7)"
        elif streak_length > 7 and streak_length <= 14:
            category = "Medium-Long (8-14)"
        elif streak_length > 14:
            category = "Long (>14)"

        add_table_row(history_table, [
            f"{int(row['streak_number'])}",
            f"{int(row['start_game_id'])} to {int(row['end_game_id'])}",
            f"{int(streak_length)}",
            category
        ])

    display_table(history_table)


def load_model_and_predict(model_path: str, latest_streaks: pd.DataFrame) -> Dict:
    """
    Load a trained model and make a prediction for the next streak.

    Args:
        model_path: Path to the saved model bundle
        latest_streaks: DataFrame with recent streak data

    Returns:
        Dictionary with prediction details
    """
    print_info(f"Loading model from {model_path}")

    try:
        model_bundle = joblib.load(model_path)
        prediction = predict_next_streak(model_bundle, latest_streaks)
        print_success("Successfully predicted next streak")
        return prediction
    except Exception as e:
        print_error(f"Error predicting next streak: {str(e)}")
        raise


def setup_prediction_service(
    model_dir: str = "./output",
    model_filename: str = "temporal_model.pkl"
) -> Dict:
    """
    Set up the prediction service by loading the model.

    Args:
        model_dir: Directory where the model is saved
        model_filename: Filename of the model

    Returns:
        Dictionary with the loaded model bundle
    """
    model_path = os.path.join(model_dir, model_filename)

    if not os.path.exists(model_path):
        print_error(f"Model file not found at {model_path}")
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print_info(f"Setting up prediction service using model at {model_path}")
    model_bundle = joblib.load(model_path)
    print_success("Model loaded successfully")

    return model_bundle
