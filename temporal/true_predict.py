#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
True prediction functionality for temporal analysis.
Ensures strict time boundaries - only past data is used for predictions.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from utils.logger_config import (
    print_info, print_success, print_warning, print_error,
    create_table, add_table_row, display_table
)


def make_true_predictions(model_bundle: Dict, streak_df: pd.DataFrame, num_streaks: int = 200) -> pd.DataFrame:
    """
    Make true forward-looking predictions using strictly temporal features with no data leakage.

    Args:
        model_bundle: Dictionary containing model and preprocessing info
        streak_df: DataFrame with all streak data
        num_streaks: Number of most recent streaks to predict

    Returns:
        DataFrame with predictions and actual values
    """
    # Extract model components
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    percentile_values = model_bundle.get('percentile_values', [3.0, 7.0, 14.0])

    # Get the most recent streaks
    if num_streaks and num_streaks < len(streak_df):
        recent_streaks = streak_df.tail(num_streaks).copy()
        print_info(
            f"Using the {num_streaks} most recent streaks for prediction")
    else:
        recent_streaks = streak_df.copy()
        print_info(f"Using all {len(streak_df)} streaks for prediction")

    # Create a list to store predictions
    predictions = []

    # Process streaks one by one in temporal order
    # Start with a sufficient history (minimum 20 streaks)
    min_history = 20
    lookback_window = 5

    print_info(
        f"Processing {len(recent_streaks)} streaks with strict temporal boundaries")

    # Create a table to display available game ranges for each prediction
    game_ranges_table = create_table(
        "Available Game Ranges For Each Prediction",
        ["Prediction #", "Streak Number",
            "Current Streak Games", "Available History Games"]
    )

    prediction_count = 0

    for i in range(min_history, len(recent_streaks)):
        # Get current streak and all previous history
        current_streak = recent_streaks.iloc[i:i+1]
        historical_streaks = recent_streaks.iloc[:i]

        # Verify no data leakage - critical assertion
        assert current_streak.iloc[0]['start_game_id'] > historical_streaks.iloc[-1]['end_game_id'], \
            f"Data leakage detected: Current streak {current_streak.iloc[0]['streak_number']} starts at " \
            f"game {current_streak.iloc[0]['start_game_id']} which is not after the end of " \
            f"the previous streak {historical_streaks.iloc[-1]['end_game_id']}"

        # Track game ranges
        current_streak_range = f"{int(current_streak.iloc[0]['start_game_id'])} to {int(current_streak.iloc[0]['end_game_id'])}"
        history_range = f"{int(historical_streaks.iloc[0]['start_game_id'])} to {int(historical_streaks.iloc[-1]['end_game_id'])}"

        # Add to table (but only display first 10 rows to avoid console spam)
        prediction_count += 1
        if prediction_count <= 10:
            add_table_row(game_ranges_table, [
                str(prediction_count),
                str(int(current_streak.iloc[0]['streak_number'])),
                current_streak_range,
                history_range
            ])

        # Create features using only historical data
        features = _create_strictly_temporal_features(
            historical_streaks,
            current_streak,
            lookback_window=lookback_window
        )

        # Extract feature values
        if not all(col in features.columns for col in feature_cols):
            missing_cols = [
                col for col in feature_cols if col not in features.columns]
            print_warning(
                f"Missing {len(missing_cols)} feature columns. Adding with default 0 values.")
            for col in missing_cols:
                features[col] = 0

        X = features[feature_cols]

        # Scale features
        X_scaled = scaler.transform(X)

        # Create DMatrix for prediction
        dpredict = xgb.DMatrix(
            X_scaled,
            feature_names=[f'f{i}' for i in range(X_scaled.shape[1])]
        )

        # Make prediction
        y_pred_proba = model.predict(dpredict)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Get actual target cluster
        streak_length = current_streak.iloc[0]['streak_length']
        target_cluster = _get_target_cluster(streak_length, percentile_values)

        # Get prediction confidence
        confidence = np.max(y_pred_proba, axis=1)[0]

        # Map to descriptive categories
        cluster_to_desc = {
            0: 'short',
            1: 'medium_short',
            2: 'medium_long',
            3: 'long'
        }

        prediction_desc = cluster_to_desc[y_pred[0]]
        actual_desc = cluster_to_desc[target_cluster]

        # Create prediction record
        prediction = {
            'streak_number': current_streak.iloc[0]['streak_number'],
            'start_game_id': current_streak.iloc[0]['start_game_id'],
            'end_game_id': current_streak.iloc[0]['end_game_id'],
            'streak_length': streak_length,
            'actual_cluster': target_cluster,
            'actual_desc': actual_desc,
            'predicted_cluster': y_pred[0],
            'prediction_desc': prediction_desc,
            'correct': (y_pred[0] == target_cluster),
            'confidence': confidence,
            'available_history_start': historical_streaks.iloc[0]['start_game_id'],
            'available_history_end': historical_streaks.iloc[-1]['end_game_id']
        }

        # Add probability for each class
        for j in range(len(y_pred_proba[0])):
            prediction[f'prob_class_{j}'] = y_pred_proba[0][j]

        predictions.append(prediction)

    # Display the game ranges table
    display_table(game_ranges_table)
    print_info(
        f"Displayed first 10 of {prediction_count} predictions. All ranges stored in output CSV.")

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame(predictions)

    # Calculate accuracy metrics
    if len(predictions_df) > 0:
        accuracy = (predictions_df['correct'].sum() /
                    len(predictions_df)) * 100
        print_info(f"Overall true prediction accuracy: {accuracy:.2f}%")

        # Display class-specific accuracy
        class_accuracy = {}
        for cluster in range(4):
            class_df = predictions_df[predictions_df['actual_cluster'] == cluster]
            if len(class_df) > 0:
                class_acc = (class_df['correct'].sum() / len(class_df)) * 100
                class_accuracy[cluster_to_desc[cluster]] = f"{class_acc:.2f}%"

        # Create a table for class accuracy
        accuracy_table = create_table(
            "True Prediction Class Accuracy", ["Class", "Accuracy"])
        for class_name, acc in class_accuracy.items():
            add_table_row(accuracy_table, [class_name, acc])
        display_table(accuracy_table)

    return predictions_df


def _create_strictly_temporal_features(
    historical_streaks: pd.DataFrame,
    current_streak: pd.DataFrame,
    lookback_window: int = 5
) -> pd.DataFrame:
    """
    Create strictly temporal features for the current streak using only historical data.

    Args:
        historical_streaks: DataFrame with historical streak data
        current_streak: DataFrame with the current streak to predict
        lookback_window: Number of previous streaks to use for features

    Returns:
        DataFrame with temporal features for the current streak
    """
    # Initialize features DataFrame with the current streak
    features = current_streak.copy()

    # Get the most recent historical streaks up to lookback_window
    recent_history = historical_streaks.tail(lookback_window).copy()

    # Create lagged features
    for i in range(1, min(lookback_window, len(recent_history)) + 1):
        if i <= len(recent_history):
            historical_idx = len(recent_history) - i
            features[f'prev{i}_length'] = recent_history.iloc[historical_idx]['streak_length']
            features[f'prev{i}_hit_mult'] = recent_history.iloc[historical_idx]['hit_multiplier']

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
        hit_mults.append(recent_history.iloc[idx]['hit_multiplier'])

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

    # Set 'same_as_prev' feature
    if len(categories) > 0:
        streak_length = features['streak_length'].iloc[0]
        current_cat = 'short'
        if streak_length > 3 and streak_length <= 7:
            current_cat = 'medium_short'
        elif streak_length > 7 and streak_length <= 14:
            current_cat = 'medium_long'
        elif streak_length > 14:
            current_cat = 'long'

        features['same_as_prev'] = 1 if current_cat == categories[0] else 0
    else:
        features['same_as_prev'] = 0

    return features


def _get_target_cluster(streak_length: int, percentile_values: List[float]) -> int:
    """
    Determine the target cluster for a given streak length.

    Args:
        streak_length: Length of the streak
        percentile_values: List of percentile values for clustering

    Returns:
        Target cluster (0, 1, 2, or 3)
    """
    if streak_length <= percentile_values[0]:
        return 0
    elif streak_length <= percentile_values[1]:
        return 1
    elif streak_length <= percentile_values[2]:
        return 2
    else:
        return 3


def analyze_true_prediction_results(predictions_df: pd.DataFrame, output_dir: str) -> None:
    """
    Analyze and display true prediction results.

    Args:
        predictions_df: DataFrame with prediction results
        output_dir: Directory to save output files
    """
    # Create a confusion matrix
    confusion_matrix = create_table(
        "True Prediction Confusion Matrix",
        ["Actual\\Predicted", "Pred: Short", "Pred: Medium-Short",
            "Pred: Medium-Long", "Pred: Long"]
    )

    # Count occurrences for each actual/predicted combination
    cm = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            cm[i, j] = len(predictions_df[(predictions_df['actual_cluster'] == i) &
                                          (predictions_df['predicted_cluster'] == j)])

    # Calculate recall for each class
    recalls = []
    for i in range(4):
        if cm[i].sum() > 0:
            recall = cm[i, i] / cm[i].sum() * 100
            recalls.append(f"{recall:.1f}%")
        else:
            recalls.append("0.0%")

    # Display confusion matrix with recalls
    class_labels = ["Short (1-3)", "Medium-Short (4-7)",
                    "Medium-Long (8-14)", "Long (>14)"]
    for i in range(4):
        row = [f"Act: {class_labels[i]}"]
        for j in range(4):
            cell_value = str(cm[i, j])
            if i == j:
                cell_value += f" ({recalls[i]} recall)"
            row.append(cell_value)
        add_table_row(confusion_matrix, row)

    display_table(confusion_matrix)

    # Display a summary of game ranges
    range_summary_table = create_table(
        "Game Range Summary",
        ["Metric", "Start Game ID", "End Game ID", "Range Size"]
    )

    # For first prediction
    first_pred = predictions_df.iloc[0]
    first_streak_range = int(
        first_pred['end_game_id']) - int(first_pred['start_game_id']) + 1
    first_history_range = int(
        first_pred['available_history_end']) - int(first_pred['available_history_start']) + 1

    add_table_row(range_summary_table, [
        "First Prediction - Current Streak",
        f"{int(first_pred['start_game_id'])}",
        f"{int(first_pred['end_game_id'])}",
        f"{first_streak_range} games"
    ])

    add_table_row(range_summary_table, [
        "First Prediction - Available History",
        f"{int(first_pred['available_history_start'])}",
        f"{int(first_pred['available_history_end'])}",
        f"{first_history_range} games"
    ])

    # For last prediction
    last_pred = predictions_df.iloc[-1]
    last_streak_range = int(
        last_pred['end_game_id']) - int(last_pred['start_game_id']) + 1
    last_history_range = int(
        last_pred['available_history_end']) - int(last_pred['available_history_start']) + 1

    add_table_row(range_summary_table, [
        "Last Prediction - Current Streak",
        f"{int(last_pred['start_game_id'])}",
        f"{int(last_pred['end_game_id'])}",
        f"{last_streak_range} games"
    ])

    add_table_row(range_summary_table, [
        "Last Prediction - Available History",
        f"{int(last_pred['available_history_start'])}",
        f"{int(last_pred['available_history_end'])}",
        f"{last_history_range} games"
    ])

    # Total range
    total_range = int(last_pred['end_game_id']) - \
        int(first_pred['available_history_start']) + 1

    add_table_row(range_summary_table, [
        "Total Game Range Covered",
        f"{int(first_pred['available_history_start'])}",
        f"{int(last_pred['end_game_id'])}",
        f"{total_range} games"
    ])

    display_table(range_summary_table)

    # Create a table for sample predictions
    sample_table = create_table(
        "Sample True Predictions",
        ["Streak #", "Start Game ID", "End Game ID", "Streak Length",
         "Available History", "Actual", "Predicted", "Confidence"]
    )

    # Add the most recent predictions
    for _, row in predictions_df.tail(10).iterrows():
        history_range = f"{int(row['available_history_start'])} to {int(row['available_history_end'])}"
        add_table_row(sample_table, [
            str(int(row['streak_number'])),
            str(int(row['start_game_id'])),
            str(int(row['end_game_id'])),
            str(int(row['streak_length'])),
            history_range,
            row['actual_desc'],
            row['prediction_desc'],
            f"{row['confidence']:.3f}"
        ])

    display_table(sample_table)

    # Save predictions to CSV
    output_path = os.path.join(output_dir, 'true_predictions.csv')
    predictions_df.to_csv(output_path, index=False)
    print_success(f"Saved true predictions to {output_path}")

    # Save detailed game ranges to a separate CSV
    game_ranges_df = predictions_df[[
        'streak_number', 'start_game_id', 'end_game_id',
        'available_history_start', 'available_history_end'
    ]].copy()

    game_ranges_df['streak_range'] = game_ranges_df.apply(
        lambda row: f"{int(row['start_game_id'])} to {int(row['end_game_id'])}",
        axis=1
    )

    game_ranges_df['history_range'] = game_ranges_df.apply(
        lambda row: f"{int(row['available_history_start'])} to {int(row['available_history_end'])}",
        axis=1
    )

    ranges_output_path = os.path.join(
        output_dir, 'true_prediction_game_ranges.csv')
    game_ranges_df.to_csv(ranges_output_path, index=False)
    print_success(f"Saved detailed game ranges to {ranges_output_path}")
