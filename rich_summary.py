#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rich Summary Display for Temporal Analysis Results.

This script displays a formatted summary of the temporal analysis outputs.
"""

import os
import json
import sys
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Import rich logging
from logger_config import (
    setup_logging, print_info, print_success, print_panel, create_stats_table,
    create_table, add_table_row, display_table
)

# Setup rich logging
logger = setup_logging()


def display_output_summary(output_dir="./output"):
    """
    Display a rich summary of temporal analysis outputs.

    Args:
        output_dir: Directory containing the output files
    """
    # Create a panel for the summary
    print_panel(
        "Temporal Analysis Output Summary",
        title="Summary",
        style="blue"
    )

    # Check if output directory exists
    if not os.path.exists(output_dir):
        print_info(f"Output directory {output_dir} does not exist.")
        return

    # List of files to check
    expected_files = [
        "temporal_model.pkl",
        "temporal_feature_importance.png",
        "temporal_performance_metrics.json",
        "accuracy_over_time.png",
        "confusion_matrix.png"
    ]

    # Check for files and create summary
    file_status = {}
    for filename in expected_files:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            modified = datetime.fromtimestamp(
                mtime).strftime('%Y-%m-%d %H:%M:%S')
            file_status[filename] = {
                "exists": True,
                "size": size,
                "modified": modified
            }
        else:
            file_status[filename] = {
                "exists": False
            }

    # Create a summary table
    summary_table = create_table(
        "Output Files", ["File", "Status", "Size", "Last Modified"])

    for filename, status in file_status.items():
        if status["exists"]:
            size_str = f"{status['size'] / 1024:.1f} KB"
            add_table_row(summary_table, [
                filename,
                "[green]Available[/green]",
                size_str,
                status["modified"]
            ])
        else:
            add_table_row(summary_table, [
                filename,
                "[red]Missing[/red]",
                "N/A",
                "N/A"
            ])

    display_table(summary_table)

    # Display feature importance from model file if available
    model_path = os.path.join(output_dir, "temporal_model.pkl")
    if os.path.exists(model_path):
        try:
            print_info("Feature Importance Summary:")
            model_bundle = joblib.load(model_path)

            if "feature_importance" in model_bundle:
                # Extract feature importance data
                importance_dict = model_bundle["feature_importance"]
                features = importance_dict.get("Feature", [])
                importance = importance_dict.get("Importance", [])
                percentage = importance_dict.get("Percentage", [])

                # Create feature importance table for top 10 features
                importance_table = create_table(
                    "Top 10 Feature Importance",
                    ["Feature", "Importance", "% of Total"]
                )

                # Get indices for top 10 features (they may not be in sorted order in the dictionary)
                if len(features) > 10:
                    # Create sorted indices based on importance
                    sorted_indices = sorted(
                        range(len(importance)),
                        key=lambda i: float(importance[i]) if isinstance(
                            importance[i], (int, float, str)) else 0,
                        reverse=True
                    )[:10]
                else:
                    sorted_indices = range(len(features))

                # Add rows for top features
                for idx in sorted_indices:
                    feature_name = features[idx] if idx < len(
                        features) else f"Feature {idx}"
                    importance_value = importance[idx] if idx < len(
                        importance) else 0
                    percentage_value = percentage[idx] if idx < len(
                        percentage) else 0

                    # Format values properly
                    if isinstance(importance_value, (int, float)):
                        importance_str = f"{float(importance_value):.2f}"
                    else:
                        importance_str = str(importance_value)

                    if isinstance(percentage_value, (int, float)):
                        percentage_str = f"{float(percentage_value):.2f}%"
                    else:
                        percentage_str = str(percentage_value)

                    add_table_row(importance_table, [
                        feature_name,
                        importance_str,
                        percentage_str
                    ])

                display_table(importance_table)

            # Also display model metrics if available
            if "metrics" in model_bundle:
                metrics = model_bundle["metrics"]
                metrics_table = create_table(
                    "Model Performance Summary",
                    ["Metric", "Value"]
                )

                # Add rows for metrics
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                        if "improvement" in metric.lower() or "percent" in metric.lower():
                            formatted_value = f"{value:.2f}%"
                    else:
                        formatted_value = str(value)

                    # Format metric name for display (convert snake_case to Title Case)
                    display_name = " ".join(word.capitalize()
                                            for word in metric.split("_"))

                    add_table_row(metrics_table, [
                        display_name,
                        formatted_value
                    ])

                display_table(metrics_table)

        except Exception as e:
            print_info(f"Could not read model file: {str(e)}")

    # If metrics file exists, show summary statistics
    metrics_file = os.path.join(
        output_dir, "temporal_performance_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            print_info("Complete Performance Metrics:")

            # Display performance by streak category
            if 'by_streak_category' in metrics:
                print_info("Performance by Streak Category:")
                category_table = create_table("Accuracy by Category",
                                              ["Category", "Accuracy"])

                for category, accuracy in metrics['by_streak_category'].items():
                    add_table_row(category_table, [
                        category,
                        f"{float(accuracy) * 100:.2f}%"
                    ])

                display_table(category_table)

            # Display performance by streak length
            if 'by_streak_length' in metrics:
                print_info("Performance by Streak Length Range:")
                length_table = create_table("Accuracy by Streak Length",
                                            ["Length Range", "Accuracy"])

                # Order the streak length ranges properly
                ordered_ranges = ["1-3", "4-7", "8-14", "15-30", "31+"]

                for length_range in ordered_ranges:
                    if length_range in metrics['by_streak_length']:
                        accuracy = metrics['by_streak_length'][length_range]
                        add_table_row(length_table, [
                            length_range,
                            f"{float(accuracy) * 100:.2f}%"
                        ])

                display_table(length_table)

            # Display performance for ALL time periods
            if 'by_time_period' in metrics:
                print_info("Performance Across All Time Periods:")

                # Get all time periods and sort them
                time_periods = list(metrics['by_time_period'].keys())
                # Sort numerically by extracting period numbers
                time_periods = sorted(
                    time_periods, key=lambda x: int(x.replace('P', '')))

                time_table = create_table("Accuracy Over Time",
                                          ["Time Period", "Accuracy"])

                for period in time_periods:
                    accuracy = metrics['by_time_period'][period]
                    add_table_row(time_table, [
                        period,
                        f"{float(accuracy) * 100:.2f}%"
                    ])

                display_table(time_table)

        except Exception as e:
            print_info(f"Could not read metrics file: {str(e)}")

    print_success("Summary display complete")


def display_custom_output_summary(output_files, title="Custom Output Summary"):
    """
    Display a rich summary of specified output files.

    Args:
        output_files: List of (name, path) tuples for output files
        title: Title for the summary panel
    """
    # Create a panel for the summary
    print_panel(
        title,
        title="Summary",
        style="blue"
    )

    # Create a summary table
    summary_table = create_table(
        "Output Files", ["File", "Status", "Size", "Last Modified"])

    for name, file_path in output_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            mtime = os.path.getmtime(file_path)
            modified = datetime.fromtimestamp(
                mtime).strftime('%Y-%m-%d %H:%M:%S')

            # Format file size for display
            size_str = f"{size / 1024:.1f} KB"

            add_table_row(summary_table, [
                name,
                "[green]Available[/green]",
                size_str,
                modified
            ])
        else:
            add_table_row(summary_table, [
                name,
                "[red]Missing[/red]",
                "N/A",
                "N/A"
            ])

    display_table(summary_table)
    print_success("Summary display complete")


def main():
    """Main function to run the summary display."""
    # Get output directory from command line if provided
    output_dir = "./output"
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

    display_output_summary(output_dir)


if __name__ == "__main__":
    main()
