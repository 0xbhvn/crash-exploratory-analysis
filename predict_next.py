#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Next streak prediction module for Crash Game 10× Streak Analysis.

This script provides a focused approach to predicting the next streak length by:
1. Loading the most recent streaks from the dataset
2. Using an ensemble of model prediction, historical patterns, and transition analysis
3. Outputting a clear, formatted prediction for the next streak only
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


def load_model(model_path: str = './output/temporal_model.pkl'):
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

        # Check if we have streak_lengths.csv for the most recent streak
        streak_lengths_path = os.path.join('./output', 'streak_lengths.csv')
        if os.path.exists(streak_lengths_path):
            try:
                streak_lengths_df = pd.read_csv(streak_lengths_path)
                if not streak_lengths_df.empty and 'streak_number' in streak_lengths_df.columns:
                    # Get the last streak from streak_lengths.csv
                    last_streak_csv = streak_lengths_df.iloc[-1]['streak_number']
                    last_streak_data = recent_streaks[-1]['streak_number']

                    if last_streak_csv > last_streak_data:
                        print_info(
                            f"Found more recent streak in streak_lengths.csv: #{last_streak_csv} vs #{last_streak_data} in games.csv")
                        print_info(
                            "Using streak_lengths.csv for the most up-to-date information")

                        # Create a panel to emphasize we're predicting a newer streak
                        print_panel(
                            f"Predicting streak #{last_streak_csv + 1} (after the most recent record in streak_lengths.csv)",
                            title="Next Streak Prediction",
                            style="yellow"
                        )
                    else:
                        # Create a panel to emphasize what we're predicting
                        next_streak_num = recent_streaks[-1]['streak_number'] + 1
                        print_panel(
                            f"Predicting streak #{next_streak_num} (after the most recent record)",
                            title="Next Streak Prediction",
                            style="yellow"
                        )
            except Exception as e:
                print_warning(f"Could not read streak_lengths.csv: {str(e)}")
                # Create a panel to emphasize what we're predicting
                next_streak_num = recent_streaks[-1]['streak_number'] + 1
                print_panel(
                    f"Predicting streak #{next_streak_num} (after the most recent record)",
                    title="Next Streak Prediction",
                    style="yellow"
                )
        else:
            # Create a panel to emphasize what we're predicting
            next_streak_num = recent_streaks[-1]['streak_number'] + 1
            print_panel(
                f"Predicting streak #{next_streak_num} (after the most recent record)",
                title="Next Streak Prediction",
                style="yellow"
            )

        # Display most recent streak information
        if recent_streaks:
            last_streak = recent_streaks[-1]
            streak_info = {
                "Most Recent Streak": f"#{last_streak['streak_number']}",
                "Game ID Range": f"{last_streak['start_game_id']} → {last_streak['end_game_id']}",
                "Streak Length": last_streak['streak_length'],
                "Hit Multiplier": f"{last_streak['hit_multiplier']:.2f}",
                "Next Streak to Predict": f"#{last_streak['streak_number'] + 1}"
            }
            create_stats_table("Recent Streak Information", streak_info)

        return recent_streaks
    except Exception as e:
        print_error(f"Error loading data: {str(e)}")
        return []


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
        f"Cluster 0: Short (1-{int(percentile_values[0])} games)",
        f"Cluster 1: Medium-Short ({int(percentile_values[0])+1}-{int(percentile_values[1])} games)",
        f"Cluster 2: Medium-Long ({int(percentile_values[1])+1}-{int(percentile_values[2])} games)",
        f"Cluster 3: Long (>{int(percentile_values[2])} games)"
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
        Dictionary with transition probabilities and last cluster
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


def predict_next_streak(model_bundle: Dict, streaks: List[Dict]):
    """
    Create a prediction for the next streak using ensemble methods.

    Args:
        model_bundle: Model bundle with required components
        streaks: List of streak dictionaries

    Returns:
        Dictionary with predicted probabilities
    """
    print_info("Generating next streak prediction")

    # Extract model components
    model = model_bundle.get("model")
    feature_cols = model_bundle.get("feature_cols", [])
    scaler = model_bundle.get("scaler")
    percentile_values = model_bundle.get("percentile_values", [3.0, 7.0, 14.0])
    percentiles = model_bundle.get("percentiles", [0.25, 0.50, 0.75])

    # 1. Get model prediction using the standard approach
    # Use a default equal probability distribution if prediction fails
    try:
        from modeling import predict_next_cluster
        model_probs = predict_next_cluster(
            model_bundle, streaks, window=50,
            feature_cols=feature_cols,
            percentiles=percentiles,
            scaler=scaler
        )
    except Exception as e:
        print_warning(f"Model prediction failed: {str(e)}")
        print_info("Using equal probability distribution for model prediction")
        model_probs = {str(i): 0.25 for i in range(4)}

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

    # Create cluster descriptions
    cluster_descriptions = {
        '0': f"Short streak (1-{int(percentile_values[0])} games)",
        '1': f"Medium-short streak ({int(percentile_values[0])+1}-{int(percentile_values[1])} games)",
        '2': f"Medium-long streak ({int(percentile_values[1])+1}-{int(percentile_values[2])} games)",
        '3': f"Long streak (>{int(percentile_values[2])} games)"
    }

    # Display a clear prediction for the next streak
    if streaks:
        next_streak_num = streaks[-1]['streak_number'] + 1

        # Sort probabilities from highest to lowest
        sorted_probs = sorted(
            balanced_probs.items(), key=lambda x: float(x[1]), reverse=True)

        # Create a final prediction table
        pred_table = create_table(
            f"Prediction for Streak #{next_streak_num}",
            ["Predicted Length", "Description", "Probability", "Rank"]
        )

        for i, (cluster, prob) in enumerate(sorted_probs):
            add_table_row(pred_table, [
                cluster,
                cluster_descriptions.get(cluster, "Unknown"),
                f"{float(prob)*100:.2f}%",
                f"#{i+1}"
            ])

        display_table(pred_table)

        # Add a final clear statement
        print_panel(
            f"For streak #{next_streak_num}, expect a {cluster_descriptions.get(sorted_probs[0][0])} with {float(sorted_probs[0][1])*100:.1f}% probability",
            title="Final Prediction",
            style="green bold"
        )

    return balanced_probs


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Next Streak Prediction for Crash Game')
    parser.add_argument('--input', default='games.csv',
                        help='Path to input CSV file with Game ID and Bust columns')
    parser.add_argument('--model_path', default='./output/temporal_model.pkl',
                        help='Path to the saved model file')
    parser.add_argument('--num_streaks', type=int, default=50,
                        help='Number of recent streaks to use for analysis')
    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for considering a multiplier as a hit (default: 10.0)')

    return parser.parse_args()


def main():
    """
    Main function to predict the next streak.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Display welcome message
    print_panel(
        f"Next Crash Game {args.multiplier_threshold}× Streak Prediction",
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

    # Predict next streak
    predict_next_streak(model_bundle, streaks)

    print_success("Next streak prediction complete!")


if __name__ == "__main__":
    main()
