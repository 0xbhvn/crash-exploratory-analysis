#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command line interface for temporal analysis.
"""

import os
import sys
import argparse
import joblib
import xgboost as xgb
from utils.logger_config import (
    setup_logging, print_info, print_success, print_error, print_panel, print_warning,
    create_table, add_table_row, display_table
)
from utils.rich_summary import display_output_summary, display_custom_output_summary

from temporal.loader import load_data
from temporal.features import create_temporal_features
from temporal.splitting import temporal_train_test_split
from temporal.training import train_temporal_model
from temporal.evaluation import analyze_temporal_performance, analyze_recall_improvements
from temporal.prediction import make_temporal_prediction
from temporal.true_predict import make_true_predictions, analyze_true_prediction_results
from temporal.deploy import load_model_and_predict


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Temporal Analysis for Crash Game 10× Streak Prediction')

    parser.add_argument('--mode', choices=['train', 'predict', 'true_predict', 'next_streak'], default='train',
                        help='Mode to run the script in (train, predict, true_predict, or next_streak)')

    parser.add_argument('--input', type=str, required=True,
                        help='Path to input CSV file with game data')

    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save output files')

    parser.add_argument('--multiplier_threshold', type=float, default=10.0,
                        help='Threshold for streak hits (default: 10.0)')

    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Size of test set for temporal validation (default: 0.2)')

    parser.add_argument('--lookback', type=int, default=5,
                        help='Number of previous streaks to use for features (default: 5)')

    parser.add_argument('--num_streaks', type=int, default=None,
                        help='Number of most recent streaks to use for prediction (prediction mode only)')

    # Create a mutually exclusive group for class balancing methods
    class_balance_group = parser.add_mutually_exclusive_group()

    class_balance_group.add_argument('--use_class_weights', action='store_true',
                                     help='Use class weights to improve recall for minority classes')

    parser.add_argument('--weight_scale', type=float, default=1.1,
                        help='Scale factor for class weights (higher values give more weight to minority classes)')

    class_balance_group.add_argument('--use_smote', action='store_true',
                                     help='Use SMOTE to generate synthetic examples for minority classes')

    parser.add_argument('--smote_k_neighbors', type=int, default=5,
                        help='Number of nearest neighbors to use for SMOTE (default: 5)')

    # Model parameters
    parser.add_argument('--max_depth', type=int, default=6,
                        help='Maximum depth of XGBoost trees (default: 6)')

    # New hyperparameters for regularization
    parser.add_argument('--eta', type=float, default=0.05,
                        help='Learning rate for XGBoost (default: 0.05, recommended range: 0.03-0.05)')

    parser.add_argument('--num_rounds', type=int, default=1000,
                        help='Maximum number of boosting rounds (default: 1000)')

    parser.add_argument('--early_stopping', type=int, default=100,
                        help='Early stopping rounds (default: 100)')

    parser.add_argument('--gamma', type=float, default=0,
                        help='Minimum loss reduction for further partition (default: 0, recommended for tuning: 0.5-1.0)')

    parser.add_argument('--min_child_weight', type=float, default=1,
                        help='Minimum sum of instance weight in child (default: 1, recommended for tuning: 1-5)')

    parser.add_argument('--reg_lambda', type=float, default=1.0,
                        help='L2 regularization (default: 1.0, recommended for tuning: 1.0-1.5)')

    parser.add_argument('--subsample', type=float, default=0.8,
                        help='Subsample ratio of training data (default: 0.8)')

    parser.add_argument('--colsample_bytree', type=float, default=0.8,
                        help='Subsample ratio of columns per tree (default: 0.8)')

    return parser.parse_args()


def train_mode(args):
    """
    Run in train mode to create and evaluate a model.

    Args:
        args: Command line arguments
    """
    # Load the data
    streak_df = load_data(args.input, args.multiplier_threshold)

    # Create temporal features
    features_df, feature_cols, percentile_values = create_temporal_features(
        streak_df, lookback_window=args.lookback)

    # Print total feature count and newly added features
    _display_feature_info(feature_cols)

    # Display model configuration
    _display_model_configuration(args)

    # Perform temporal train-test split
    X_train, y_train, X_test, y_test, test_indices = temporal_train_test_split(
        features_df, feature_cols, test_size=args.test_size)

    # Print class balancing strategy information
    _display_class_balancing_strategy(args)

    # Train temporal model
    model, model_bundle = train_temporal_model(
        X_train, y_train, X_test, y_test, feature_cols, args.output_dir,
        use_class_weights=args.use_class_weights,
        weight_scale=args.weight_scale,
        use_smote=args.use_smote,
        smote_k_neighbors=args.smote_k_neighbors,
        max_depth=args.max_depth,
        eta=args.eta,
        num_rounds=args.num_rounds,
        early_stopping=args.early_stopping,
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree)

    # Get predictions on test set
    X_test_scaled = model_bundle['scaler'].transform(X_test)
    dtest = joblib.load(os.path.join(args.output_dir, 'temporal_model.pkl'))['model'].predict(
        xgb.DMatrix(X_test_scaled, feature_names=[
                    f'f{i}' for i in range(X_test_scaled.shape[1])])
    )
    y_pred = model_bundle['predictions']

    # Analyze recall improvements if using class weighting methods
    if args.use_class_weights or args.use_smote:
        report = analyze_recall_improvements(y_test, y_pred, args.output_dir)

    # Analyze temporal performance
    analyze_temporal_performance(
        features_df, y_pred, test_indices, percentile_values, args.output_dir)

    # Display rich summary of outputs
    display_output_summary(args.output_dir)


def predict_mode(args):
    """
    Run in predict mode to make predictions on new data.

    Args:
        args: Command line arguments
    """
    # Load the model bundle
    model_path = os.path.join(args.output_dir, 'temporal_model.pkl')

    if not os.path.exists(model_path):
        print_error("Model file not found. Train a model first.")
        sys.exit(1)

    print_info(f"Loading model from {model_path}")
    model_bundle = joblib.load(model_path)

    # Load the data
    streak_df = load_data(args.input, args.multiplier_threshold)

    # Use the most recent streaks for prediction
    recent_streaks = _get_recent_streaks(streak_df, args.num_streaks)

    # Make predictions
    prediction_df = make_temporal_prediction(
        model_bundle, recent_streaks,
        temporal_idx_start=len(streak_df)-len(recent_streaks)
    )

    # Display prediction distribution
    _display_prediction_distribution(prediction_df)

    # Display sample predictions
    _display_sample_predictions(prediction_df)

    # Analyze predictions if ground truth is available
    if 'target_cluster' in prediction_df.columns:
        _analyze_predictions_with_ground_truth(prediction_df, args.output_dir)

    # Save predictions to CSV
    output_path = os.path.join(args.output_dir, 'temporal_predictions.csv')
    prediction_df.to_csv(output_path, index=False)
    print_success(f"Saved predictions to {output_path}")

    # Display confidence statistics
    _display_confidence_statistics(prediction_df)

    # Display output files summary
    _display_prediction_output_summary(
        output_path, prediction_df, args.output_dir)

    print_success("Prediction analysis complete!")


def true_predict_mode(args):
    """
    Run in true prediction mode to make forward-only predictions with no data leakage.

    Args:
        args: Command line arguments
    """
    # Load the model bundle
    model_path = os.path.join(args.output_dir, 'temporal_model.pkl')

    if not os.path.exists(model_path):
        print_error("Model file not found. Train a model first.")
        sys.exit(1)

    print_info(f"Loading model from {model_path}")
    model_bundle = joblib.load(model_path)

    # Load the data
    streak_df = load_data(args.input, args.multiplier_threshold)

    # Display welcome panel for true prediction mode
    print_panel(
        "True Forward-Looking Prediction Mode\n"
        "This mode ensures strict temporal boundaries with no data leakage.\n"
        "Each streak is predicted using only data from previous streaks.",
        title="True Prediction Mode",
        style="green bold"
    )

    # Make true predictions
    print_info("Creating true predictions with strict temporal boundaries")
    predictions_df = make_true_predictions(
        model_bundle,
        streak_df,
        num_streaks=args.num_streaks
    )

    # Analyze prediction results
    analyze_true_prediction_results(predictions_df, args.output_dir)

    # Display confidence statistics
    _display_confidence_statistics(predictions_df)

    # Display output files summary
    output_path = os.path.join(args.output_dir, 'true_predictions.csv')
    _display_prediction_output_summary(
        output_path, predictions_df, args.output_dir)

    print_success("True prediction analysis complete!")


def next_streak_mode(args):
    """
    Run in next streak mode to predict the streak after the latest available data.

    Args:
        args: Command line arguments
    """
    # Load the model bundle
    model_path = os.path.join(args.output_dir, 'temporal_model.pkl')

    if not os.path.exists(model_path):
        print_error("Model file not found. Train a model first.")
        sys.exit(1)

    print_info(f"Loading model from {model_path}")

    # Load the data
    streak_df = load_data(args.input, args.multiplier_threshold)

    # Display welcome panel for next streak prediction mode
    print_panel(
        "Next Streak Prediction Mode\n"
        "This mode predicts the upcoming streak after the most recent data.\n"
        "Only historical data is used with no data leakage.",
        title="Next Streak Prediction",
        style="green bold"
    )

    # Use the most recent streaks for prediction
    if args.num_streaks and args.num_streaks < len(streak_df):
        recent_streaks = streak_df.tail(args.num_streaks).copy()
        print_info(
            f"Using the {args.num_streaks} most recent streaks for prediction")
    else:
        recent_streaks = streak_df.copy()
        print_info(
            f"Using all {len(streak_df)} streaks for prediction base")

    # Make prediction for next streak
    prediction = load_model_and_predict(model_path, recent_streaks)

    # Save prediction to JSON
    import json
    output_path = os.path.join(args.output_dir, 'next_streak_prediction.json')
    with open(output_path, 'w') as f:
        json.dump(prediction, f, indent=4)

    print_success(f"Saved next streak prediction to {output_path}")
    print_success("Next streak prediction complete!")


def _display_feature_info(feature_cols):
    """
    Display information about the features.

    Args:
        feature_cols: List of feature column names
    """
    # Print newly added features
    new_features = [
        col for col in feature_cols
        if col.startswith('hit_mult_') or col.startswith('time_since_')
    ]

    if len(new_features) > 0:
        print_info(
            f"Added {len(new_features)} new features: {', '.join(new_features[:5])}...")


def _display_model_configuration(args):
    """
    Display model configuration.

    Args:
        args: Command line arguments
    """
    print_panel(
        f"Model Configuration:\n"
        f"- Max Tree Depth: {args.max_depth}\n"
        f"- Learning Rate (eta): {args.eta}\n"
        f"- Max Rounds: {args.num_rounds} (early stop: {args.early_stopping})\n"
        f"- Regularization: gamma={args.gamma}, lambda={args.reg_lambda}, min_child_weight={args.min_child_weight}\n"
        f"- Sampling: subsample={args.subsample}, colsample_bytree={args.colsample_bytree}\n"
        f"- Class Weights: {'Yes' if args.use_class_weights else 'No'}\n"
        f"- Weight Scale: {args.weight_scale if args.use_class_weights else 'N/A'}\n"
        f"- SMOTE: {'Yes' if args.use_smote else 'No'}",
        title="Configuration",
        style="cyan"
    )


def _display_class_balancing_strategy(args):
    """
    Display information about the class balancing strategy.

    Args:
        args: Command line arguments
    """
    if args.use_class_weights:
        print_info(
            f"Class balancing strategy: Using class weights (scale={args.weight_scale})")
    elif args.use_smote:
        print_info(
            f"Class balancing strategy: Using SMOTE (k_neighbors={args.smote_k_neighbors})")
    else:
        print_info(
            "Class balancing strategy: None (using original class distribution)")


def _get_recent_streaks(streak_df, num_streaks):
    """
    Get the most recent streaks for prediction.

    Args:
        streak_df: DataFrame with all streaks
        num_streaks: Number of most recent streaks to use

    Returns:
        DataFrame with recent streaks
    """
    if num_streaks:
        recent_streaks = streak_df.tail(num_streaks)
        print_info(
            f"Using the {num_streaks} most recent streaks for prediction")
    else:
        recent_streaks = streak_df
        print_info(
            f"Using all {len(streak_df)} streaks for prediction analysis")

    return recent_streaks


def _display_prediction_distribution(prediction_df):
    """
    Display the distribution of predictions.

    Args:
        prediction_df: DataFrame with predictions
    """
    # Calculate prediction statistics
    pred_counts = prediction_df['prediction_desc'].value_counts()
    pred_pcts = pred_counts / len(prediction_df) * 100

    # Display prediction distribution
    pred_table = create_table("Prediction Distribution",
                              ["Predicted Length", "Count", "Percentage"])

    for cat in sorted(pred_counts.index):
        add_table_row(pred_table, [
            cat,
            f"{pred_counts[cat]}",
            f"{pred_pcts[cat]:.1f}%"
        ])

    display_table(pred_table)


def _display_sample_predictions(prediction_df):
    """
    Display sample predictions.

    Args:
        prediction_df: DataFrame with predictions
    """
    # Display sample predictions with game IDs and streak information
    sample_table = create_table("Sample Predictions",
                                ["Streak #", "Start Game ID", "End Game ID", "Streak Length", "Predicted", "Confidence"])

    # Show the last 10 predictions
    for _, row in prediction_df.tail(10).iterrows():
        add_table_row(sample_table, [
            f"{row['streak_number']}",
            f"{row['start_game_id']}",
            f"{row['end_game_id']}",
            f"{row['streak_length']}",
            f"{row['prediction_desc']}",
            f"{row['prediction_confidence']:.3f}"
        ])

    display_table(sample_table)


def _analyze_predictions_with_ground_truth(prediction_df, output_dir):
    """
    Analyze predictions with ground truth data.

    Args:
        prediction_df: DataFrame with predictions
        output_dir: Directory to save output files
    """
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report

    print_info("Creating confusion matrix for predictions")

    cm = confusion_matrix(prediction_df['target_cluster'],
                          prediction_df['predicted_cluster'])

    # Define class labels for better readability
    class_labels = [
        "Short (1-3)",
        "Medium-Short (4-7)",
        "Medium-Long (8-14)",
        "Long (>14)"
    ]

    # Display confusion matrix as rich table
    cm_table = create_table("Prediction Confusion Matrix",
                            ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

    # Calculate row totals for recall calculation
    row_totals = cm.sum(axis=1)

    # Add rows with counts and percentages
    for i, (row, label) in enumerate(zip(cm, class_labels)):
        row_data = [f"Act {i}: {label}"]
        for j, count in enumerate(row):
            recall = (count / row_totals[i]) * \
                100 if row_totals[i] > 0 else 0
            if i == j:  # Diagonal element - show recall percentage
                cell_text = f"{count} ({recall:.1f}% recall)"
            else:
                cell_text = f"{count}"
            row_data.append(cell_text)
        add_table_row(cm_table, row_data)

    display_table(cm_table)

    # Plot and save confusion matrix with percentages
    plt.figure(figsize=(10, 8))

    # Create annotation texts with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = (count / row_totals[i]) * 100 if row_totals[i] > 0 else 0
            annot[i, j] = f'{count}\n({pct:.1f}%)'

    # Plot heatmap with raw counts for colors
    ax = sns.heatmap(
        cm,  # Use raw count matrix for colors
        annot=annot,  # Use our custom annotation with both values
        fmt='',  # Empty format since we're providing formatted strings
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={'size': 11, 'va': 'center'},  # Removed bold weight
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Prediction Confusion Matrix')

    # Save confusion matrix plot
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'prediction_confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=120)
    plt.close()

    print_info(f"Saved prediction confusion matrix to {cm_path}")

    # Add precision confusion matrix
    _create_precision_confusion_matrix(
        prediction_df, cm, class_labels, output_dir)

    # Add classification report
    _create_classification_report(prediction_df, output_dir)


def _create_precision_confusion_matrix(prediction_df, cm, class_labels, output_dir):
    """
    Create a precision-focused confusion matrix visualization.

    Args:
        prediction_df: DataFrame with predictions
        cm: Confusion matrix
        class_labels: List of class labels
        output_dir: Directory to save output files
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    print_info("Creating precision-focused confusion matrix")

    # Calculate column totals for precision calculation
    col_totals = cm.sum(axis=0)

    # Create precision matrix table
    precision_table = create_table("Prediction Confusion Matrix (Precision)",
                                   ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

    # Add rows with counts and precision percentages
    for i, (row, label) in enumerate(zip(cm, class_labels)):
        row_data = [f"Act {i}: {label}"]
        for j, count in enumerate(row):
            precision = (count / col_totals[j]) * \
                100 if col_totals[j] > 0 else 0
            if i == j:  # Diagonal element - show precision percentage
                cell_text = f"{count} ({precision:.1f}% precision)"
            else:
                cell_text = f"{count}"
            row_data.append(cell_text)
        add_table_row(precision_table, row_data)

    # Add total row showing column totals
    total_row = ["Total Predictions"]
    for j, col_total in enumerate(col_totals):
        # Diagonal element is correct predictions
        correct = cm[j, j]
        precision = (correct / col_total) * 100 if col_total > 0 else 0
        total_row.append(f"{col_total} ({precision:.1f}% overall)")

    add_table_row(precision_table, total_row)
    display_table(precision_table)

    # Plot and save the precision confusion matrix - handle empty columns
    plt.figure(figsize=(10, 8))

    # Create annotation array with both count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            if col_totals[j] == 0:
                # Show 0 for empty columns instead of N/A
                annot[i, j] = f'0\n(0.0%)'
            else:
                pct = (count / col_totals[j]) * 100
                annot[i, j] = f'{count}\n({pct:.1f}%)'

    # Create the figure with custom annotations and raw count for colors
    ax = sns.heatmap(
        cm,  # Use raw count matrix for colors
        annot=annot,  # Use custom annotation that shows both values
        fmt='',  # Empty format since we're providing formatted strings
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        annot_kws={'size': 11, 'va': 'center'},  # Removed bold weight
        cbar_kws={'label': 'Count'}
    )

    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Precision Confusion Matrix')

    # Save precision confusion matrix plot
    precision_cm_path = os.path.join(
        output_dir, 'prediction_precision_matrix.png')
    plt.savefig(precision_cm_path, bbox_inches='tight', dpi=120)
    plt.close()

    print_info(f"Saved precision confusion matrix to {precision_cm_path}")


def _create_classification_report(prediction_df, output_dir):
    """
    Create a detailed classification report for predictions.

    Args:
        prediction_df: DataFrame with predictions
        output_dir: Directory to save output files
    """
    import json
    from sklearn.metrics import classification_report

    print_info("Generating classification report")
    report = classification_report(prediction_df['target_cluster'],
                                   prediction_df['predicted_cluster'],
                                   output_dict=True,
                                   zero_division=0)

    # Create a classification report table
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

    # Add accuracy row
    if 'accuracy' in report:
        add_table_row(report_table, [
            "Accuracy",
            "",
            "",
            f"{report['accuracy']:.4f}",
            f"{sum([report[cls]['support'] for cls in report if cls not in ['accuracy', 'macro avg', 'weighted avg']])}"
        ])

    display_table(report_table)

    # Save classification report to JSON
    report_path = os.path.join(
        output_dir, 'prediction_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print_info(f"Saved classification report to {report_path}")


def _display_confidence_statistics(prediction_df):
    """
    Display statistics about prediction confidence.

    Args:
        prediction_df: DataFrame with prediction results
    """
    # Check if we have confidence values
    confidence_col = 'prediction_confidence'
    if confidence_col not in prediction_df.columns and 'confidence' in prediction_df.columns:
        confidence_col = 'confidence'

    if confidence_col in prediction_df.columns:
        conf_mean = prediction_df[confidence_col].mean()
        conf_median = prediction_df[confidence_col].median()

        conf_table = create_table("Prediction Confidence", ["Metric", "Value"])
        add_table_row(conf_table, ["Mean Confidence", f"{conf_mean:.4f}"])
        add_table_row(conf_table, ["Median Confidence", f"{conf_median:.4f}"])
        display_table(conf_table)
    else:
        print_warning("No confidence values found in prediction results.")


def _display_prediction_output_summary(output_path, prediction_df, output_dir):
    """
    Display a summary of prediction outputs.

    Args:
        output_path: Path to output file
        prediction_df: DataFrame with prediction results
        output_dir: Directory containing output files
    """
    import pandas as pd

    # Create a panel for summary
    print_panel(
        "Prediction Output Files",
        title="Summary",
        style="blue"
    )

    # Create a summary table
    summary_table = create_table(
        "Output Files", ["File", "Status", "Path"])

    # Check for prediction CSV file
    if os.path.exists(output_path):
        add_table_row(summary_table, [
            "Predictions CSV",
            "[green]Available[/green]",
            output_path
        ])
    else:
        add_table_row(summary_table, [
            "Predictions CSV",
            "[red]Missing[/red]",
            output_path
        ])

    # Check for confusion matrix image
    cm_path = os.path.join(output_dir, 'prediction_confusion_matrix.png')
    if os.path.exists(cm_path):
        add_table_row(summary_table, [
            "Confusion Matrix",
            "[green]Available[/green]",
            cm_path
        ])
    else:
        add_table_row(summary_table, [
            "Confusion Matrix",
            "[red]Missing[/red]",
            cm_path
        ])

    # Check for classification report
    report_path = os.path.join(
        output_dir, 'prediction_classification_report.json')
    if os.path.exists(report_path):
        add_table_row(summary_table, [
            "Classification Report",
            "[green]Available[/green]",
            report_path
        ])
    else:
        add_table_row(summary_table, [
            "Classification Report",
            "[red]Missing[/red]",
            report_path
        ])

    display_table(summary_table)


def main():
    """
    Main function for the temporal analysis script.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Display welcome message
    print_panel(
        f"Temporal Analysis for Crash Game {args.multiplier_threshold}× Streak Prediction",
        title="Welcome",
        style="blue"
    )

    # Run in the specified mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'predict':
        predict_mode(args)
    elif args.mode == 'true_predict':
        true_predict_mode(args)
    elif args.mode == 'next_streak':
        next_streak_mode(args)
    else:
        print_error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    print_success(f"{args.mode.capitalize()} analysis complete!")


if __name__ == "__main__":
    main()
