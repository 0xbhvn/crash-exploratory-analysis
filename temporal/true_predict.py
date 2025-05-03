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
from rich.progress import Progress
# Import the main feature creation function
from temporal.features import create_temporal_features


def make_true_predictions(model_bundle: Dict, streak_df: pd.DataFrame, num_streaks: int = 200) -> pd.DataFrame:
    """
    Make true forward-looking predictions using strictly temporal features with no data leakage.

    Args:
        model_bundle: Dictionary containing model and preprocessing info
        streak_df: DataFrame with all streak data
        num_streaks: Number of most recent streaks to predict (or None for all possible)

    Returns:
        DataFrame with predictions and actual values
    """
    # Extract model components
    model = model_bundle['model']  # This is the CalibratedClassifierCV
    scaler = model_bundle['scaler']
    feature_cols = model_bundle['feature_cols']
    percentile_values = model_bundle.get('percentile_values', [3.0, 7.0, 14.0])

    # Use all streaks for potential history, filtering happens in the loop
    all_streaks = streak_df.copy()

    # Determine streaks to iterate over for prediction
    if num_streaks and num_streaks < len(all_streaks):
        start_index = len(all_streaks) - num_streaks
        print_info(
            f"Processing the {num_streaks} most recent streaks for true prediction (indices {start_index} to {len(all_streaks)-1})")
    else:
        start_index = 20  # Default min_history
        print_info(
            f"Processing all possible streaks for true prediction (indices {start_index} to {len(all_streaks)-1})")

    # Create a list to store predictions
    predictions = []

    # Process streaks one by one in temporal order
    min_history = 20  # Minimum streaks needed before first prediction
    # TODO: Make this consistent or parameterize? Using 5 for now.
    lookback_window = 5

    # Create a table to display available game ranges for each prediction
    game_ranges_table = create_table(
        "Available Game Ranges For Each Prediction",
        ["Prediction #", "Streak Number",
            "Predicted Streak Games", "Available History Games"]
    )
    prediction_count = 0
    total_streaks_to_process = len(all_streaks) - start_index

    # Ensure streaks are sorted temporally if not already
    all_streaks = all_streaks.sort_values(
        'temporal_idx').reset_index(drop=True)

    # --- Add Progress Bar ---
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Processing streaks...", total=total_streaks_to_process)

        for i in range(start_index, len(all_streaks)):
            # Streak i is the one we want to predict (the 'current' streak)
            current_streak_info = all_streaks.iloc[i]

            # History is everything *before* streak i
            historical_streaks = all_streaks.iloc[:i]

            # Ensure minimum history is met
            if len(historical_streaks) < min_history:
                # Still advance progress even if skipping
                progress.update(task, advance=1)
                continue

            # Verify no data leakage
            assert current_streak_info['start_game_id'] > historical_streaks.iloc[-1]['end_game_id'], \
                f"Data leakage detected at streak {current_streak_info['streak_number']}"

            # === Feature Generation using imported function ===
            # Call with verbose=False to suppress logs inside the loop
            features_df_hist, _, _ = create_temporal_features(
                historical_streaks,
                lookback_window=lookback_window,
                verbose=False
            )

            if features_df_hist.empty:
                print_warning(
                    f"Feature generation returned empty DataFrame for history before streak {current_streak_info['streak_number']}. Skipping prediction.")
                progress.update(task, advance=1)  # Advance progress
                continue

            # Get the features corresponding to the end of the historical period (last row)
            features_for_pred_row = features_df_hist.iloc[[-1]]

            # Handle missing columns (though ideally create_temporal_features is consistent)
            missing_cols = [
                col for col in feature_cols if col not in features_for_pred_row.columns]
            if missing_cols:
                print_warning(
                    f"Missing features for streak {current_streak_info['streak_number']}: {missing_cols}. Adding defaults.")
                for col in missing_cols:
                    # Use .loc to avoid warning
                    features_for_pred_row.loc[:, col] = 0

            # Ensure columns are in the correct order
            X_predict = features_for_pred_row[feature_cols]
            # === End Feature Generation ===

            # Scale features
            X_scaled = scaler.transform(X_predict)

            # Make prediction using the calibrated model
            y_pred_proba = model.predict_proba(X_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)[
                0]  # Get single prediction
            confidence = np.max(y_pred_proba, axis=1)[0]

            # Get actual target cluster for the current streak
            streak_length = current_streak_info['streak_length']
            target_cluster = _get_target_cluster(
                streak_length, percentile_values)

            # Map to descriptive categories
            cluster_to_desc = {
                0: 'short',
                1: 'medium_short',
                2: 'medium_long',
                3: 'long'
            }
            prediction_desc = cluster_to_desc[y_pred]
            actual_desc = cluster_to_desc[target_cluster]

            # Create prediction record
            prediction = {
                'streak_number': current_streak_info['streak_number'],
                'start_game_id': current_streak_info['start_game_id'],
                'end_game_id': current_streak_info['end_game_id'],
                'streak_length': streak_length,
                'actual_cluster': target_cluster,
                'actual_desc': actual_desc,
                'predicted_cluster': y_pred,
                'prediction_desc': prediction_desc,
                'correct': (y_pred == target_cluster),
                'confidence': confidence,
                'available_history_start': historical_streaks.iloc[0]['start_game_id'],
                'available_history_end': historical_streaks.iloc[-1]['end_game_id']
            }
            for j in range(len(y_pred_proba[0])):
                prediction[f'prob_class_{j}'] = y_pred_proba[0][j]

            predictions.append(prediction)

            # Update and display game ranges table periodically
            prediction_count += 1
            if prediction_count <= 10 or prediction_count % 1000 == 0:  # Show first 10 and then every 1000
                current_streak_range = f"{int(current_streak_info['start_game_id'])} to {int(current_streak_info['end_game_id'])}"
                history_range = f"{int(historical_streaks.iloc[0]['start_game_id'])} to {int(historical_streaks.iloc[-1]['end_game_id'])}"
                # Clear previous table rows if needed (not directly supported by rich table?)
                # For simplicity, just add rows. Table might get long in logs.
                if prediction_count <= 10:
                    add_table_row(game_ranges_table, [
                        str(prediction_count),
                        str(int(current_streak_info['streak_number'])),
                        current_streak_range,
                        history_range
                    ])
                if prediction_count == 10:  # Display initial table
                    display_table(game_ranges_table)

            # Update progress bar
            progress.update(task, advance=1)
            # --- End Progress Bar Context ---

    # Display final count message if not already shown
    if prediction_count > 10:
        print_info(
            f"Processed {prediction_count} total predictions. Final ranges stored in output CSV.")
    elif prediction_count <= 10 and prediction_count > 0:
        # Display if less than 10 predictions were made
        display_table(game_ranges_table)

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
    confusion_matrix_table = create_table(  # Renamed variable to avoid conflict
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
        add_table_row(confusion_matrix_table, row)

    display_table(confusion_matrix_table)

    # Display a summary of game ranges
    range_summary_table = create_table(
        "Game Range Summary",
        ["Metric", "Start Game ID", "End Game ID", "Range Size"]
    )

    # Handle case where predictions_df might be empty
    if not predictions_df.empty:
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

    # Add the most recent predictions (handle empty case)
    # Show up to 10
    for _, row in predictions_df.tail(min(10, len(predictions_df))).iterrows():
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

    if not game_ranges_df.empty:
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
