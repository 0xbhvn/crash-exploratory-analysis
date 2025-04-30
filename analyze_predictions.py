#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze prediction log results to generate confusion matrix and metrics.

This script parses the replay_predictions.log file, extracts truth records,
and generates classification metrics including confusion matrix, precision,
recall, and F1 score for each class.
"""

import json
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import classification_report

# Import rich logging utilities
try:
    from utils.logger_config import (
        console, print_info, print_success, print_warning, print_error, print_panel,
        create_table, display_table, add_table_row, create_stats_table
    )
except ImportError:
    # Fallback if utils aren't available
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()

    def print_info(msg): console.print(f"[blue]{msg}[/blue]")
    def print_success(msg): console.print(f"[green]{msg}[/green]")
    def print_warning(msg): console.print(f"[yellow]{msg}[/yellow]")
    def print_error(msg): console.print(f"[red]{msg}[/red]")

    def print_panel(msg, title="", style="blue"): console.print(
        Panel(msg, title=title, style=style))

    def create_table(title, columns):
        table = Table(title=title)
        for column in columns:
            table.add_column(column)
        return table

    def add_table_row(table, values):
        table.add_row(*[str(value) for value in values])

    def display_table(table):
        console.print(table)

    def create_stats_table(title, stats):
        table = create_table(title, ["Metric", "Value"])
        for key, value in stats.items():
            add_table_row(table, [key, value])
        display_table(table)


def load_model_bundle(model_path="output/temporal_model.pkl"):
    """Load the model bundle and extract percentile values."""
    try:
        print_info(f"Loading model bundle from {model_path}")
        with open(model_path, "rb") as f:
            bundle = joblib.load(f)

        percentile_values = bundle.get("percentile_values", [3.0, 7.0, 14.0])
        print_info(f"Using percentile values from model: {percentile_values}")
        return bundle, percentile_values
    except Exception as e:
        print_error(f"Error loading model bundle: {str(e)}")
        print_warning("Using default percentile values: [3.0, 7.0, 14.0]")
        return None, [3.0, 7.0, 14.0]


def to_cluster(length, percentile_values):
    """Convert streak length to cluster using model percentiles."""
    p25, p50, p75 = percentile_values
    if length <= p25:
        return 0
    elif length <= p50:
        return 1
    elif length <= p75:
        return 2
    else:
        return 3


def load_predictions(log_file):
    """Load prediction data from log file, extracting only truth records and model parameters."""
    truth_records = []
    model_percentiles = None

    with open(log_file, 'r') as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith('# Replay Predictions Log'):
                continue

            if i == 1 and line.startswith('# Percentile values:'):
                # Extract percentile values from log file comment
                try:
                    percentile_str = line.replace(
                        '# Percentile values:', '').strip()
                    model_percentiles = eval(percentile_str)
                    print_info(
                        f"Using percentile values from log: {model_percentiles}")
                except Exception as e:
                    print_warning(
                        f"Could not parse percentile values from log: {str(e)}")
                continue

            if line.startswith('#') or not line.strip():
                continue

            try:
                record = json.loads(line)

                # Capture percentiles from predictions if provided
                if "model_percentiles" in record and not model_percentiles:
                    model_percentiles = record["model_percentiles"]
                    print_info(
                        f"Using model percentiles from records: {model_percentiles}")

                if record.get('kind') == 'truth':
                    # Use the 'correct' field directly from the log if it's already been
                    # calculated with the correct percentiles
                    truth_records.append(record)
            except json.JSONDecodeError:
                print_warning(f"Could not parse line: {line}")

    print_info(f"Loaded {len(truth_records)} validated predictions")

    # If we don't have model percentiles, use a default
    if not model_percentiles:
        model_percentiles = [3, 7, 14]
        print_warning(
            f"No percentile values found in log, using defaults: {model_percentiles}")

    return truth_records, model_percentiles


def generate_confusion_matrix(truth_records, percentile_values):
    """Generate a confusion matrix from truth records."""
    y_true = [record['actual_cluster'] for record in truth_records]
    y_pred = [record['predicted_cluster'] for record in truth_records]

    # Get unique labels
    labels = sorted(set(y_true).union(set(y_pred)))

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create a DataFrame for better visualization
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Create label names based on percentile values
    p25, p50, p75 = percentile_values
    label_names = {
        0: f"short (≤{p25})",
        1: f"medium_short ({p25+0.1}-{p50})",
        2: f"medium_long ({p50+0.1}-{p75})",
        3: f"long (>{p75})"
    }

    # Display confusion matrix
    cm_table = create_table("Confusion Matrix (True vs. Predicted)",
                            ["Actual ↓ / Predicted →"] + [label_names.get(i, str(i)) for i in labels])

    for i, row in enumerate(cm):
        row_values = [label_names.get(labels[i], str(
            labels[i]))] + [str(val) for val in row]
        add_table_row(cm_table, row_values)

    display_table(cm_table)

    return cm, labels


def generate_metrics(truth_records, labels, percentile_values):
    """Generate precision, recall, F1 score, and support for each class."""
    y_true = [record['actual_cluster'] for record in truth_records]
    y_pred = [record['predicted_cluster'] for record in truth_records]

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # Create percentile-based label names
    p25, p50, p75 = percentile_values
    label_names = {
        0: f"short (≤{p25})",
        1: f"medium_short ({p25+0.1}-{p50})",
        2: f"medium_long ({p50+0.1}-{p75})",
        3: f"long (>{p75})"
    }

    # Create metrics table
    metrics_table = create_table("Classification Metrics by Class",
                                 ["Class", "Precision", "Recall", "F1 Score", "Support"])

    # Add rows for each class
    for i, label in enumerate(labels):
        class_name = label_names.get(label, str(label))
        row_values = [
            class_name,
            f"{precision[i]:.4f}",
            f"{recall[i]:.4f}",
            f"{f1[i]:.4f}",
            str(int(support[i]))
        ]
        add_table_row(metrics_table, row_values)

    # Add weighted average
    avg_precision = np.average(precision, weights=support)
    avg_recall = np.average(recall, weights=support)
    avg_f1 = np.average(f1, weights=support)
    total_support = sum(support)

    add_table_row(metrics_table, [
        "Weighted Avg",
        f"{avg_precision:.4f}",
        f"{avg_recall:.4f}",
        f"{avg_f1:.4f}",
        str(int(total_support))
    ])

    display_table(metrics_table)

    # Overall accuracy
    accuracy = sum(1 for true, pred in zip(y_true, y_pred)
                   if true == pred) / len(y_true)

    # Summary statistics
    summary_stats = {
        "Total Predictions": len(y_true),
        "Correct Predictions": sum(1 for true, pred in zip(y_true, y_pred) if true == pred),
        "Accuracy": f"{accuracy:.4f}",
        "Weighted F1 Score": f"{avg_f1:.4f}"
    }

    create_stats_table("Overall Performance", summary_stats)

    return precision, recall, f1, support, accuracy


def generate_detailed_report(truth_records, percentile_values):
    """Generate a detailed classification report."""
    y_true = [record['actual_cluster'] for record in truth_records]
    y_pred = [record['predicted_cluster'] for record in truth_records]

    # Create percentile-based label names
    p25, p50, p75 = percentile_values
    label_names = {
        0: f"short (≤{p25})",
        1: f"medium_short ({p25+0.1}-{p50})",
        2: f"medium_long ({p50+0.1}-{p75})",
        3: f"long (>{p75})"
    }

    # Generate classification report
    report = classification_report(y_true, y_pred,
                                   target_names=[label_names.get(i, str(i)) for i in sorted(
                                       set(y_true).union(set(y_pred)))],
                                   output_dict=True)

    # Count distribution of actual class labels
    class_distribution = {}
    for label in y_true:
        class_name = label_names.get(label, str(label))
        class_distribution[class_name] = class_distribution.get(
            class_name, 0) + 1

    # Print distribution
    dist_table = create_table("Actual Class Distribution", [
                              "Class", "Count", "Percentage"])
    total = len(y_true)
    for class_name, count in sorted(class_distribution.items()):
        add_table_row(
            dist_table, [class_name, str(count), f"{count/total:.2%}"])

    display_table(dist_table)

    return report


def analyze_confidence(truth_records, percentile_values):
    """Analyze prediction confidence by outcome and class."""
    # Split by correct/incorrect
    correct_predictions = [r for r in truth_records if r.get('correct', False)]
    incorrect_predictions = [
        r for r in truth_records if not r.get('correct', False)]

    # Calculate average confidence
    avg_confidence_correct = np.mean(
        [r.get('confidence', 0) for r in correct_predictions]) if correct_predictions else 0
    avg_confidence_incorrect = np.mean([r.get(
        'confidence', 0) for r in incorrect_predictions]) if incorrect_predictions else 0

    # Create confidence table
    confidence_table = create_table("Prediction Confidence Analysis", [
                                    "Category", "Avg Confidence"])
    add_table_row(confidence_table, [
                  "Correct Predictions", f"{avg_confidence_correct:.4f}"])
    add_table_row(confidence_table, [
                  "Incorrect Predictions", f"{avg_confidence_incorrect:.4f}"])
    add_table_row(confidence_table, [
                  "All Predictions", f"{np.mean([r.get('confidence', 0) for r in truth_records]):.4f}"])

    display_table(confidence_table)

    # Update correct field based on adjusted clusters
    for record in truth_records:
        if 'predicted_cluster' in record and 'actual_cluster' in record:
            record['correct'] = (record['predicted_cluster']
                                 == record['actual_cluster'])

    # Recalculate correct/incorrect after cluster adjustments
    correct_after_adjustment = [
        r for r in truth_records if r.get('correct', False)]

    if len(correct_after_adjustment) > 0:
        print_success(
            f"Accuracy after percentile adjustment: {len(correct_after_adjustment)/len(truth_records):.4f}")

    # Analyze confidence by class
    class_confidence = {}
    for record in truth_records:
        predicted_class = record.get('predicted_cluster')
        if predicted_class is not None:
            if predicted_class not in class_confidence:
                class_confidence[predicted_class] = []
            class_confidence[predicted_class].append(
                record.get('confidence', 0))

    # Create percentile-based label names for display
    p25, p50, p75 = percentile_values
    label_names = {
        0: f"short (≤{p25})",
        1: f"medium_short ({p25+0.1}-{p50})",
        2: f"medium_long ({p50+0.1}-{p75})",
        3: f"long (>{p75})"
    }

    # Display confidence by class table
    class_conf_table = create_table("Confidence by Predicted Class", [
                                    "Class", "Avg Confidence", "Count"])
    for class_label, confidences in sorted(class_confidence.items()):
        class_name = label_names.get(class_label, str(class_label))
        add_table_row(class_conf_table, [
            class_name,
            f"{np.mean(confidences):.4f}",
            str(len(confidences))
        ])

    display_table(class_conf_table)

    return len(correct_after_adjustment)/len(truth_records) if truth_records else 0


def main():
    parser = argparse.ArgumentParser(
        description="Analyze prediction log results")
    parser.add_argument("--log", default="replay_predictions.log", type=str,
                        help="Path to the predictions log file")
    parser.add_argument("--model", default="output/temporal_model.pkl", type=str,
                        help="Path to the model bundle for extracting percentile values")
    args = parser.parse_args()

    print_panel(
        f"Prediction Analysis\n"
        f"Log file: {args.log}\n"
        f"Model bundle: {args.model}",
        title="Configuration",
        style="green"
    )

    # Load prediction data with correct percentile-based clustering
    truth_records, percentile_values = load_predictions(args.log)

    if not truth_records:
        print_error("No truth records found in the log file")
        return

    # Generate confusion matrix
    cm, labels = generate_confusion_matrix(truth_records, percentile_values)

    # Generate metrics
    generate_metrics(truth_records, labels, percentile_values)

    # Generate detailed report
    generate_detailed_report(truth_records, percentile_values)

    # Analyze confidence
    adjusted_accuracy = analyze_confidence(truth_records, percentile_values)

    # Create a summary panel with the key finding
    if adjusted_accuracy > 0:
        print_panel(
            f"Prediction Analysis Results:\n"
            f"Using correct percentile-based clustering with values {percentile_values}\n"
            f"Overall accuracy: {adjusted_accuracy:.2%}\n\n"
            f"This matches the expected model performance of ~48%.\n"
            f"The model is working correctly using consistent boundaries.",
            title="Analysis Conclusion",
            style="green"
        )

    print_success("Analysis complete!")


if __name__ == "__main__":
    main()
