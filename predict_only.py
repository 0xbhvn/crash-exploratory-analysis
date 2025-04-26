#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Balanced prediction module for Crash Game 10× Streak Analysis.

This script provides a more balanced approach to streak prediction by:
1. Focusing more on historical patterns than current streak properties
2. Using a simple ensemble approach with different feature sets
3. Providing more realistic probability estimates
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from typing import Dict, List, Tuple, Optional

# Import rich logging
from logger_config import (
    setup_logging, console, print_info, print_success, print_warning,
    print_error, print_panel, create_stats_table, create_table, add_table_row,
    display_table
)

# Import from local modules
from data_processing import extract_streaks_and_multipliers

# Setup rich logging
logger = setup_logging()


def load_model(model_path: str = './output/xgboost_model.pkl'):
    """
    Load the prediction model bundle.

    Args:
        model_path: Path to the saved model bundle

    Returns:
        Model bundle dictionary or None if loading fails
    """
    print_info(f"Loading model from {model_path}")
    try:
        model_bundle = joblib.load(model_path)
        # Extract key components
        if isinstance(model_bundle, dict):
            model = model_bundle.get("model")
            feature_cols = model_bundle.get("feature_cols", [])
            scaler = model_bundle.get("scaler")
            percentile_values = model_bundle.get(
                "percentile_values", [3.0, 7.0, 14.0])
            percentiles = model_bundle.get("percentiles", [0.25, 0.50, 0.75])

            print_info(
                f"Successfully loaded model with {len(feature_cols)} features")

            # Create and display a summary of the model
            model_info = {
                "Model Type": "XGBoost",
                "Features": len(feature_cols),
                "Clusters": len(percentiles) + 1,
                "Percentile Boundaries": ", ".join([f"{p*100:.0f}%" for p in percentiles]),
                "Streak Boundaries": ", ".join([f"{int(v)}" for v in percentile_values])
            }

            create_stats_table("Model Information", model_info)

            return model_bundle
        else:
            print_error("Invalid model bundle format")
            return None
    except Exception as e:
        print_error(f"Error loading model: {str(e)}")
        return None


def load_recent_streaks(csv_path: str, num_streaks: int = 50, multiplier_threshold: float = 10.0):
    """
    Load the most recent streaks from the CSV file.

    Args:
        csv_path: Path to the CSV file with game data
        num_streaks: Number of recent streaks to extract
        multiplier_threshold: Threshold for streak hits

    Returns:
        List of dictionaries with streak information
    """
    print_info(f"Loading most recent data from {csv_path}")
    try:
        # Load data
        df = pd.read_csv(csv_path)
        df["Game ID"] = df["Game ID"].astype(int)
        df["Bust"] = df["Bust"].astype(float)

        print_info(
            f"Loaded {len(df)} games, extracting the most recent {num_streaks} streaks")

        # Extract streaks
        all_streaks = extract_streaks_and_multipliers(df, multiplier_threshold)

        # Get the most recent streaks
        recent_streaks = all_streaks.tail(num_streaks).to_dict('records')

        # Display streak range information
        if recent_streaks:
            first_streak = recent_streaks[0]
            last_streak = recent_streaks[-1]
            streak_range = f"Streaks #{first_streak['streak_number']} → #{last_streak['streak_number']}"
            print_info(f"Streak range: {streak_range}")

            game_range = f"Games #{first_streak['start_game_id']} → #{last_streak['end_game_id']}"
            print_info(f"Game ID range: {game_range}")

        # Display streak summary table
        streaks_table = create_table(
            "Input Streaks for Prediction",
            ["Streak #", "Game ID Range", "Length", "Hit Multiplier"]
        )

        for i, streak in enumerate(recent_streaks):
            streak_label = f"Streak #{streak['streak_number']}"
            if i == len(recent_streaks) - 1:
                streak_label += " (most recent)"

            game_range = f"{streak['start_game_id']} → {streak['end_game_id']}"

            add_table_row(streaks_table, [
                streak_label,
                game_range,
                streak['streak_length'],
                f"{streak['hit_multiplier']:.2f}"
            ])

        # Add a row for the predicted streak
        if recent_streaks:
            next_streak_id = recent_streaks[-1]['streak_number'] + 1
            add_table_row(streaks_table, [
                f"Streak #{next_streak_id} (predicted)",
                "? → ?",
                "?",
                "?"
            ])

        display_table(streaks_table)

        return recent_streaks
    except Exception as e:
        print_error(f"Error loading data: {str(e)}")
        return []


def create_lagged_features(streaks: List[Dict], window: int = 50):
    """
    Create lagged features focusing only on streak lengths and pattern detection.

    Args:
        streaks: List of streak dictionaries
        window: Window size for feature creation

    Returns:
        Pandas DataFrame with lagged features
    """
    print_info(
        f"Creating pattern-focused features from {len(streaks)} streaks")

    # Create a DataFrame from the streaks
    streak_df = pd.DataFrame(streaks)

    # Initialize feature dictionaries
    lagged_features = {}
    rolling_features = {}
    pattern_features = {}

    # Create lagged features - focus on streak lengths
    for i in range(1, min(window, len(streaks)) + 1):
        lagged_features[f'prev{i}_length'] = streak_df['streak_length'].shift(
            i)

    # Create rolling window statistics
    rolling_features['rolling_mean_5'] = streak_df['streak_length'].shift(
        1).rolling(5, min_periods=1).mean()
    rolling_features['rolling_mean_10'] = streak_df['streak_length'].shift(
        1).rolling(10, min_periods=1).mean()
    rolling_features['rolling_mean_20'] = streak_df['streak_length'].shift(
        1).rolling(20, min_periods=1).mean()

    rolling_features['rolling_std_5'] = streak_df['streak_length'].shift(
        1).rolling(5, min_periods=2).std().fillna(0)
    rolling_features['rolling_std_10'] = streak_df['streak_length'].shift(
        1).rolling(10, min_periods=2).std().fillna(0)

    # Add streak length category features (short, medium, long)
    pattern_features['prev_short'] = (
        streak_df['streak_length'].shift(1) <= 3).astype(int)
    pattern_features['prev_medium'] = ((streak_df['streak_length'].shift(1) > 3) &
                                       (streak_df['streak_length'].shift(1) <= 14)).astype(int)
    pattern_features['prev_long'] = (
        streak_df['streak_length'].shift(1) > 14).astype(int)

    # Count recent streaks by category (last 5)
    for i in range(2, 6):
        pattern_features[f'short_in_last_{i}'] = 0
        pattern_features[f'medium_in_last_{i}'] = 0
        pattern_features[f'long_in_last_{i}'] = 0

    for i in range(5):
        if i+1 < len(streaks):
            for j in range(2, 6):
                if i < j:
                    length = streak_df['streak_length'].iloc[-(i+1)]
                    if length <= 3:
                        pattern_features[f'short_in_last_{j}'].iloc[-1] += 1
                    elif length <= 14:
                        pattern_features[f'medium_in_last_{j}'].iloc[-1] += 1
                    else:
                        pattern_features[f'long_in_last_{j}'].iloc[-1] += 1

    # Combine all features
    lagged_df = pd.DataFrame(lagged_features, index=streak_df.index)
    rolling_df = pd.DataFrame(rolling_features, index=streak_df.index)
    pattern_df = pd.DataFrame(pattern_features, index=streak_df.index)

    features_df = pd.concat(
        [streak_df, lagged_df, rolling_df, pattern_df], axis=1)

    # Get the last row which contains features for prediction
    if not features_df.empty:
        last_features = features_df.iloc[-1]
        print_info(f"Created {len(last_features)} features for prediction")

        # Print some key features
        if 'rolling_mean_5' in last_features:
            print_info(
                f"Last 5 streak average: {last_features['rolling_mean_5']:.2f}")
        if 'rolling_mean_10' in last_features:
            print_info(
                f"Last 10 streak average: {last_features['rolling_mean_10']:.2f}")

        # Print pattern counts
        print_info(
            f"Last streak category: {'short' if last_features['prev_short'] else 'medium' if last_features['prev_medium'] else 'long'}")

        short_count = last_features.get('short_in_last_5', 0)
        medium_count = last_features.get('medium_in_last_5', 0)
        long_count = last_features.get('long_in_last_5', 0)
        print_info(
            f"Last 5 streaks: {short_count} short, {medium_count} medium, {long_count} long")

    return features_df


def calculate_historical_probabilities(streaks: List[Dict], percentile_values: List[float]):
    """
    Calculate historical probabilities based on streak patterns.

    Args:
        streaks: List of streak dictionaries
        percentile_values: List of percentile values for clustering

    Returns:
        Dictionary with probabilities for each cluster
    """
    print_info("Calculating historical probabilities")

    # Count streaks by cluster
    counts = [0, 0, 0, 0]  # Initialize counts for 4 clusters
    total = 0

    # Group streaks by clusters
    for streak in streaks:
        length = streak['streak_length']
        total += 1

        if length <= percentile_values[0]:
            counts[0] += 1
        elif length <= percentile_values[1]:
            counts[1] += 1
        elif length <= percentile_values[2]:
            counts[2] += 1
        else:
            counts[3] += 1

    # Calculate probabilities
    probs = {str(i): count/total for i, count in enumerate(counts)}

    # Create a table to display the historical distribution
    hist_table = create_table("Historical Distribution", [
                              "Cluster", "Count", "Percentage"])

    cluster_labels = [
        f"Cluster 0: Bottom 25% (1-{int(percentile_values[0])} streak length)",
        f"Cluster 1: 25-50% ({int(percentile_values[0])+1}-{int(percentile_values[1])} streak length)",
        f"Cluster 2: 50-75% ({int(percentile_values[1])+1}-{int(percentile_values[2])} streak length)",
        f"Cluster 3: Top 25% (>{int(percentile_values[2])} streak length)"
    ]

    for i, label in enumerate(cluster_labels):
        add_table_row(hist_table, [
            label,
            counts[i],
            f"{counts[i]/total*100:.2f}%"
        ])

    display_table(hist_table)

    return probs


def analyze_transitions(streaks: List[Dict], percentile_values: List[float]):
    """
    Analyze transitions between streak clusters to detect patterns.

    Args:
        streaks: List of streak dictionaries
        percentile_values: List of percentile values for clustering

    Returns:
        Dictionary with transition probabilities
    """
    print_info("Analyzing cluster transitions")

    # Identify cluster for each streak
    clusters = []
    for streak in streaks:
        length = streak['streak_length']

        if length <= percentile_values[0]:
            clusters.append(0)
        elif length <= percentile_values[1]:
            clusters.append(1)
        elif length <= percentile_values[2]:
            clusters.append(2)
        else:
            clusters.append(3)

    # Calculate transitions
    transitions = {
        '0': {'0': 0, '1': 0, '2': 0, '3': 0},
        '1': {'0': 0, '1': 0, '2': 0, '3': 0},
        '2': {'0': 0, '1': 0, '2': 0, '3': 0},
        '3': {'0': 0, '1': 0, '2': 0, '3': 0}
    }

    transition_counts = {
        '0': 0, '1': 0, '2': 0, '3': 0
    }

    # Count transitions
    for i in range(len(clusters) - 1):
        current = str(clusters[i])
        next_cluster = str(clusters[i + 1])

        transitions[current][next_cluster] += 1
        transition_counts[current] += 1

    # Calculate probabilities
    transition_probs = {}
    for current in transitions:
        if transition_counts[current] > 0:
            transition_probs[current] = {
                next_cluster: count / transition_counts[current]
                for next_cluster, count in transitions[current].items()
            }
        else:
            transition_probs[current] = {
                next_cluster: 0.25  # Equal probability if no data
                for next_cluster in transitions[current]
            }

    # Display transition probabilities
    trans_table = create_table("Transition Probabilities",
                               ["From Cluster", "To Cluster 0", "To Cluster 1", "To Cluster 2", "To Cluster 3"])

    for current in sorted(transition_probs.keys()):
        probs = transition_probs[current]
        add_table_row(trans_table, [
            f"Cluster {current}",
            f"{probs['0']*100:.2f}%",
            f"{probs['1']*100:.2f}%",
            f"{probs['2']*100:.2f}%",
            f"{probs['3']*100:.2f}%"
        ])

    display_table(trans_table)

    # Get the last cluster
    last_cluster = str(clusters[-1]) if clusters else '0'

    return transition_probs, last_cluster


def balanced_prediction(model_bundle: Dict, streaks: List[Dict]):
    """
    Create a more balanced prediction using ensemble methods.

    Args:
        model_bundle: Model bundle with required components
        streaks: List of streak dictionaries

    Returns:
        Dictionary with balanced probabilities
    """
    print_info("Generating balanced streak prediction")

    # Extract model components
    model = model_bundle.get("model")
    feature_cols = model_bundle.get("feature_cols", [])
    scaler = model_bundle.get("scaler")
    percentile_values = model_bundle.get("percentile_values", [3.0, 7.0, 14.0])
    percentiles = model_bundle.get("percentiles", [0.25, 0.50, 0.75])

    # 1. Get model prediction using the standard approach
    from modeling import predict_next_cluster
    model_probs = predict_next_cluster(
        model_bundle, streaks, window=50,
        feature_cols=feature_cols,
        percentiles=percentiles,
        scaler=scaler
    )

    # 2. Calculate historical distribution
    hist_probs = calculate_historical_probabilities(streaks, percentile_values)

    # 3. Calculate transition probabilities
    transition_probs, last_cluster = analyze_transitions(
        streaks, percentile_values)
    trans_prediction = transition_probs[last_cluster]

    # 4. Create a simple ensemble prediction
    # Weights: 25% model, 25% historical, 50% transition-based
    balanced_probs = {}

    # Print model prediction table
    pred_table = create_table("Prediction Components",
                              ["Cluster", "Model Prediction", "Historical", "Transition", "Balanced"])

    for cluster in model_probs:
        # Calculate ensemble prediction
        balanced_probs[cluster] = (
            0.25 * float(model_probs[cluster]) +
            0.25 * float(hist_probs[cluster]) +
            0.5 * float(trans_prediction[cluster])
        )

        # Add to table
        add_table_row(pred_table, [
            f"Cluster {cluster}",
            f"{float(model_probs[cluster])*100:.2f}%",
            f"{float(hist_probs[cluster])*100:.2f}%",
            f"{float(trans_prediction[cluster])*100:.2f}%",
            f"{balanced_probs[cluster]*100:.2f}%"
        ])

    display_table(pred_table)

    # Sort by probability (descending)
    sorted_results = sorted(
        balanced_probs.items(), key=lambda x: float(x[1]), reverse=True)

    # Display results in a table
    prediction_table = create_table(
        "Balanced Streak Prediction Results", ["Cluster", "Description", "Probability"])

    # Generate cluster descriptions based on percentiles
    cluster_descriptions = {
        '0': f"Cluster 0: Bottom 25% (1-{int(percentile_values[0])} streak length)",
        '1': f"Cluster 1: 25-50% ({int(percentile_values[0])+1}-{int(percentile_values[1])} streak length)",
        '2': f"Cluster 2: 50-75% ({int(percentile_values[1])+1}-{int(percentile_values[2])} streak length)",
        '3': f"Cluster 3: Top 25% (>{int(percentile_values[2])} streak length)"
    }

    for cluster, prob in sorted_results:
        description = cluster_descriptions.get(cluster, "Unknown")
        add_table_row(prediction_table, [
            cluster, description, f"{float(prob) * 100:.2f}%"])

    display_table(prediction_table)

    # Determine the most likely streak length range based on highest probability
    if sorted_results:
        most_likely_cluster = sorted_results[0][0]
        most_likely_desc = cluster_descriptions.get(
            most_likely_cluster, "Unknown")
        most_likely_prob = sorted_results[0][1] * 100

        # Create a rich panel with prediction summary
        prediction_summary = (
            f"Most likely outcome: {most_likely_desc} ({most_likely_prob:.1f}% probability)\n"
            f"Top two probabilities: {sorted_results[0][0]}: {sorted_results[0][1]*100:.1f}%, {sorted_results[1][0]}: {sorted_results[1][1]*100:.1f}%\n"
        )

        # Add next streak number if available
        if streaks and 'streak_number' in streaks[-1]:
            next_streak_num = streaks[-1]['streak_number'] + 1
            prediction_summary += f"Predicting for: Streak #{next_streak_num}"

        print_panel(prediction_summary,
                    title="Balanced Streak Prediction", style="green")

    return balanced_probs


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Balanced Crash Game Streak Prediction')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--model_path', default='./output/xgboost_model.pkl',
                        help='Path to the saved model file')
    parser.add_argument('--num_streaks', type=int, default=50,
                        help='Number of recent streaks to analyze')
    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for considering a multiplier as a hit (default: 10.0)')

    return parser.parse_args()


def main():
    """
    Main function to run the balanced prediction.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Display welcome message
    print_panel(
        f"Balanced Crash Game {args.multiplier_threshold}× Streak Prediction",
        title="Welcome",
        style="blue"
    )

    # Load model
    model_bundle = load_model(args.model_path)
    if not model_bundle:
        print_error("Failed to load model. Exiting.")
        sys.exit(1)

    # Load recent streaks
    streaks = load_recent_streaks(
        args.input,
        args.num_streaks,
        args.multiplier_threshold
    )
    if not streaks:
        print_error("Failed to load streak data. Exiting.")
        sys.exit(1)

    # Create balanced prediction
    balanced_probs = balanced_prediction(model_bundle, streaks)

    print_success("Balanced streak prediction complete!")


if __name__ == "__main__":
    main()
