#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rich Summary Display for Temporal Analysis Results.

This script displays a formatted summary of the temporal analysis outputs.
"""

import os
import json
import sys
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
        "accuracy_over_time.png"
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

    # If metrics file exists, show summary statistics
    metrics_file = os.path.join(
        output_dir, "temporal_performance_metrics.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

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

            # Display performance by time period (last 3 periods)
            if 'by_time_period' in metrics:
                print_info("Performance in Recent Time Periods:")

                # Get the last 3 time periods
                time_periods = list(metrics['by_time_period'].keys())
                time_periods.sort()
                recent_periods = time_periods[-3:]

                time_table = create_table("Recent Accuracy Trends",
                                          ["Time Period", "Accuracy"])

                for period in recent_periods:
                    accuracy = metrics['by_time_period'][period]
                    add_table_row(time_table, [
                        period,
                        f"{float(accuracy) * 100:.2f}%"
                    ])

                display_table(time_table)

        except Exception as e:
            print_info(f"Could not read metrics file: {str(e)}")

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
