#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model evaluation functionality for temporal analysis.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from sklearn.metrics import classification_report, confusion_matrix
from utils.logger_config import (
    print_info, print_warning, print_success, create_table, add_table_row, display_table
)


def analyze_temporal_performance(features_df, y_pred, test_indices, percentile_values, output_dir):
    """
    Analyze the model's performance across different temporal patterns.

    Args:
        features_df: DataFrame with all features
        y_pred: Predicted labels
        test_indices: Indices of the test set
        percentile_values: Percentile values used for clustering
        output_dir: Directory to save outputs

    Returns:
        Dictionary of performance metrics
    """
    print_info("Analyzing temporal prediction performance")

    # Create test set with predictions
    test_df = features_df.loc[test_indices].copy()
    test_df['predicted_cluster'] = y_pred
    test_df['correct'] = (test_df['predicted_cluster'] ==
                          test_df['target_cluster']).astype(int)

    # Analyze class distribution
    _analyze_class_distribution(test_df)

    # Analyze performance by categories
    performance_metrics = _analyze_performance_by_categories(
        test_df, percentile_values)

    # Analyze performance over time
    time_perf = _analyze_performance_over_time(test_df)
    performance_metrics['by_time_period'] = time_perf.to_dict()

    # Save performance metrics
    metrics_path = _save_performance_metrics(performance_metrics, output_dir)

    # Create visualizations
    _create_time_performance_plot(time_perf, output_dir)
    _create_confusion_matrix(test_df, output_dir)

    print_info(f"Saved temporal performance metrics to {metrics_path}")

    return performance_metrics


def _analyze_class_distribution(test_df):
    """
    Analyze the distribution of actual and predicted classes.

    Args:
        test_df: DataFrame with test data and predictions
    """
    # Get counts for actual classes
    actual_counts = test_df['target_cluster'].value_counts().sort_index()
    actual_pcts = actual_counts / len(test_df) * 100

    # Get counts for predicted classes
    pred_counts = test_df['predicted_cluster'].value_counts().sort_index()
    pred_pcts = pred_counts / len(test_df) * 100

    # Create a metrics table
    metrics_table = create_table("Class Distribution Metrics",
                                 ["Class", "Actual Count", "Actual %", "Predicted Count", "Predicted %"])

    # Add rows for each class
    for cls in sorted(set(test_df['target_cluster'].unique()).union(test_df['predicted_cluster'].unique())):
        act_count = actual_counts.get(cls, 0)
        act_pct = actual_pcts.get(cls, 0)
        pred_count = pred_counts.get(cls, 0)
        pred_pct = pred_pcts.get(cls, 0)

        # Map class index to description
        if cls == 0:
            cls_desc = f"{cls}: Bottom 25% (1-3)"
        elif cls == 1:
            cls_desc = f"{cls}: 25-50% (4-7)"
        elif cls == 2:
            cls_desc = f"{cls}: 50-75% (8-14)"
        elif cls == 3:
            cls_desc = f"{cls}: Top 25% (>14)"
        else:
            cls_desc = f"Class {cls}"

        add_table_row(metrics_table, [
            cls_desc,
            f"{act_count}",
            f"{act_pct:.2f}%",
            f"{pred_count}",
            f"{pred_pct:.2f}%"
        ])

    # Add total row
    add_table_row(metrics_table, [
        "Total",
        f"{len(test_df)}",
        "100.00%",
        f"{len(test_df)}",
        "100.00%"
    ])

    display_table(metrics_table)


def _analyze_performance_by_categories(test_df, percentile_values):
    """
    Analyze model performance by different streak categories.

    Args:
        test_df: DataFrame with test data and predictions
        percentile_values: Percentile values used for clustering

    Returns:
        Dictionary with performance metrics by category
    """
    # Performance metrics dictionary
    performance_metrics = {}

    # Analyze performance by streak category (if available)
    if 'streak_category' in test_df.columns:
        category_perf = test_df.groupby('streak_category', observed=False)[
            'correct'].mean()
        performance_metrics['by_streak_category'] = category_perf.to_dict()

    # Analyze performance by streak length ranges
    test_df['streak_length_range'] = pd.cut(
        test_df['streak_length'],
        bins=[0, 3, 7, 14, 30, float('inf')],
        labels=['1-3', '4-7', '8-14', '15-30', '31+']
    )
    length_perf = test_df.groupby('streak_length_range', observed=False)[
        'correct'].mean()
    performance_metrics['by_streak_length'] = length_perf.to_dict()

    # Add streak categories to test set based on percentiles
    test_df['streak_category_name'] = pd.cut(
        test_df['streak_length'],
        bins=[0] + percentile_values + [float('inf')],
        labels=['short', 'medium_short', 'medium_long', 'long']
    )

    # Add one-hot encoded streak categories
    for cat in test_df['streak_category_name'].cat.categories:
        test_df[f'is_{cat}'] = (
            test_df['streak_category_name'] == cat).astype(int)

    # Calculate transition probabilities if available
    if 'streak_category' in test_df.columns:
        # Create previous category column if it doesn't exist
        if 'prev_category' not in test_df.columns:
            test_df['prev_category'] = test_df['streak_category'].shift(1)

        # Calculate performance by streak category and previous category
        streak_cat_perf = test_df.groupby(
            ['streak_category', 'prev_category'], observed=False)['correct'].mean()
        performance_metrics['by_category_transition'] = streak_cat_perf.to_dict()

        # Calculate transition probabilities
        transitions = pd.crosstab(
            test_df['streak_category'],
            test_df['prev_category'],
            normalize='index'
        )
        performance_metrics['transition_matrix'] = transitions.to_dict()

    return performance_metrics


def _analyze_performance_over_time(test_df):
    """
    Analyze model performance over time periods.

    Args:
        test_df: DataFrame with test data and predictions

    Returns:
        Series with performance by time period
    """
    # Create temporal groups
    test_df['temporal_group'] = pd.qcut(
        test_df['temporal_idx'],
        q=10,
        labels=[f"P{i+1}" for i in range(10)]
    )

    # Calculate performance by time group
    time_perf = test_df.groupby('temporal_group', observed=False)[
        'correct'].mean()

    # Sort time periods numerically instead of lexicographically
    time_perf = time_perf.reset_index()
    # Extract period numbers and convert to integers for proper sorting
    time_perf['period_num'] = time_perf['temporal_group'].str.extract(
        r'P(\d+)').astype(int)
    time_perf = time_perf.sort_values('period_num')
    # Convert back to a series with correct ordering
    time_perf = pd.Series(
        time_perf['correct'].values, index=time_perf['temporal_group'].values)

    return time_perf


def _save_performance_metrics(performance_metrics, output_dir):
    """
    Save performance metrics to a JSON file.

    Args:
        performance_metrics: Dictionary of performance metrics
        output_dir: Directory to save the metrics

    Returns:
        Path to the saved metrics file
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(
        output_dir, 'temporal_performance_metrics.json')

    # Convert dictionary keys to strings to ensure JSON compatibility
    json_compatible_metrics = {}
    for key, value in performance_metrics.items():
        # Convert tuple keys to strings
        if isinstance(key, tuple):
            new_key = str(key)
        else:
            new_key = key

        # Handle nested dictionaries
        if isinstance(value, dict):
            new_value = {}
            for k, v in value.items():
                if isinstance(k, tuple):
                    new_value[str(k)] = v
                else:
                    new_value[k] = v
            json_compatible_metrics[new_key] = new_value
        else:
            json_compatible_metrics[new_key] = value

    with open(metrics_path, 'w') as f:
        json.dump(json_compatible_metrics, f, indent=4, default=str)

    return metrics_path


def _create_time_performance_plot(time_perf, output_dir):
    """
    Create and save a plot of performance over time.

    Args:
        time_perf: Series with performance by time period
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    time_perf.plot(kind='bar')
    plt.title('Prediction Accuracy Over Time')
    plt.ylabel('Accuracy')
    plt.xlabel('Time Period')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save time performance plot
    plot_path = os.path.join(output_dir, 'accuracy_over_time.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved temporal performance plot to {plot_path}")


def _create_confusion_matrix(test_df, output_dir):
    """
    Create and save a confusion matrix visualization.

    Args:
        test_df: DataFrame with test data and predictions
        output_dir: Directory to save the confusion matrix
    """
    print_info("Creating confusion matrix visualization")

    cm = confusion_matrix(test_df['target_cluster'],
                          test_df['predicted_cluster'])

    # Define class labels for better readability
    class_labels = [
        "Bottom 25% (1-3)",
        "25-50% (4-7)",
        "50-75% (8-14)",
        "Top 25% (>14)"
    ]

    # Create the confusion matrix table
    _display_confusion_matrix_table(cm, class_labels)

    # Create the confusion matrix plot
    _plot_confusion_matrix(cm, class_labels, output_dir)


def _display_confusion_matrix_table(cm, class_labels):
    """
    Display the confusion matrix as a rich table.

    Args:
        cm: Confusion matrix array
        class_labels: List of class labels
    """
    # Display confusion matrix as rich table
    print_info("Confusion Matrix Table:")
    cm_table = create_table("Confusion Matrix",
                            ["Actual\\Predicted"] + [f"Pred {i}: {label}" for i, label in enumerate(class_labels)])

    # Calculate column totals for prediction accuracy percentages
    col_totals = cm.sum(axis=0)
    total_sum = cm.sum()

    # Add rows with counts and percentages
    for i, (row, label) in enumerate(zip(cm, class_labels)):
        row_data = [f"Act {i}: {label}"]
        for j, count in enumerate(row):
            if i == j:  # Diagonal element - show prediction accuracy
                prediction_accuracy = (
                    count / col_totals[j]) * 100 if col_totals[j] > 0 else 0
                cell_text = f"{count} ({prediction_accuracy:.1f}% correct)"
            else:
                cell_text = f"{count}"
            row_data.append(cell_text)
        add_table_row(cm_table, row_data)

    # Add a totals row
    total_row = ["Total Predictions"]
    for col_total in col_totals:
        percentage = (col_total / total_sum) * 100
        total_row.append(f"{col_total} ({percentage:.1f}%)")
    add_table_row(cm_table, total_row)

    display_table(cm_table)


def _plot_confusion_matrix(cm, class_labels, output_dir):
    """
    Plot and save the confusion matrix.

    Args:
        cm: Confusion matrix array
        class_labels: List of class labels
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')

    # Save confusion matrix plot
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved confusion matrix visualization to {cm_path}")


def analyze_recall_improvements(y_test, y_pred, output_dir):
    """
    Analyze and visualize recall improvements, especially for long streaks.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_dir: Directory to save outputs

    Returns:
        Classification report dictionary
    """
    print_info("Analyzing recall improvements by class")

    # Get the classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extract metrics
    metrics = _extract_classification_metrics(report)

    # Analyze distribution balance
    has_skew = _analyze_prediction_balance(y_test, y_pred)

    # Create visualization
    _create_recall_visualization(metrics, y_test, y_pred, output_dir)

    # Analyze key insights
    _analyze_key_recall_insights(metrics)

    # Generate recommendations
    _generate_recommendations(metrics, has_skew)

    return report


def _extract_classification_metrics(report):
    """
    Extract metrics from the classification report.

    Args:
        report: Classification report dictionary

    Returns:
        Dictionary with extracted metrics
    """
    # Extract metrics for each class
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for cls in sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]):
        classes.append(f"Class {cls}")
        precisions.append(report[cls]['precision'])
        recalls.append(report[cls]['recall'])
        f1_scores.append(report[cls]['f1-score'])
        supports.append(report[cls]['support'])

    return {
        'classes': classes,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'supports': supports
    }


def _analyze_prediction_balance(y_test, y_pred):
    """
    Analyze the balance between actual and predicted class distributions.

    Args:
        y_test: True labels
        y_pred: Predicted labels

    Returns:
        Boolean indicating if there's significant skew
    """
    # Calculate distribution metrics
    actual_counts = pd.Series(y_test).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    # Calculate percentages
    total_actual = len(y_test)
    total_pred = len(y_pred)
    actual_pcts = actual_counts / total_actual * 100
    pred_pcts = pred_counts / total_pred * 100

    # Check for prediction skew
    skew_table = create_table("Prediction Balance Check",
                              ["Class", "Actual %", "Predicted %", "Difference", "Status"])

    has_skew = False
    for cls in sorted(set(actual_counts.index.union(pred_counts.index))):
        act_pct = actual_pcts.get(cls, 0)
        pred_pct = pred_pcts.get(cls, 0)
        diff = pred_pct - act_pct
        abs_diff = abs(diff)

        # Determine status
        if abs_diff < 5:
            status = "✓ Good"
        elif abs_diff < 10:
            status = "⚠ Fair"
        else:
            status = "✗ Skewed"
            has_skew = True

        add_table_row(skew_table, [
            f"Class {cls}",
            f"{act_pct:.1f}%",
            f"{pred_pct:.1f}%",
            f"{diff:+.1f}%",
            status
        ])

    display_table(skew_table)

    if has_skew:
        print_warning(
            "Model predictions show class imbalance compared to actual distribution.")
        print_info(
            "Consider adjusting class weights or using a different balancing method.")
    else:
        print_success(
            "Model predictions match the actual class distribution well.")

    return has_skew


def _create_recall_visualization(metrics, y_test, y_pred, output_dir):
    """
    Create and save visualization of precision, recall, and F1-score by class.

    Args:
        metrics: Dictionary with extracted metrics
        y_test: True labels
        y_pred: Predicted labels
        output_dir: Directory to save the plot
    """
    # Create a comparison bar chart
    plt.figure(figsize=(12, 8))
    width = 0.25
    x = np.arange(len(metrics['classes']))

    plt.bar(x - width, metrics['precisions'], width, label='Precision')
    plt.bar(x, metrics['recalls'], width, label='Recall')
    plt.bar(x + width, metrics['f1_scores'], width, label='F1-Score')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-Score by Class')
    plt.xticks(x, metrics['classes'])
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data labels
    for i, v in enumerate(metrics['precisions']):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(metrics['recalls']):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    for i, v in enumerate(metrics['f1_scores']):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center')

    # Add class descriptions at the bottom
    class_descriptions = [
        "Short (1-3)",
        "Medium-Short (4-7)",
        "Medium-Long (8-14)",
        "Long (>14)"
    ]

    for i, desc in enumerate(class_descriptions):
        plt.text(i, -0.05, desc, ha='center', fontsize=9)

    # Add the actual vs predicted class distribution
    _add_distribution_inset(plt, metrics['classes'], y_test, y_pred)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'recall_improvements.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print_info(f"Saved recall improvement visualization to {plot_path}")


def _add_distribution_inset(plt, classes, y_test, y_pred):
    """
    Add an inset axes showing actual vs predicted class distribution.

    Args:
        plt: Matplotlib pyplot
        classes: List of class names
        y_test: True labels
        y_pred: Predicted labels
    """
    distribution_ax = plt.axes([0.15, 0.15, 0.3, 0.2])  # inset axes
    dist_x = np.arange(len(classes))
    dist_width = 0.35

    # Get actual and predicted percentages
    actual_counts = pd.Series(y_test).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    total_actual = len(y_test)
    total_pred = len(y_pred)

    act_pct_values = []
    pred_pct_values = []

    for cls in range(len(classes)):
        act_pct_values.append(
            (actual_counts.get(float(cls), 0) / total_actual) * 100)
        pred_pct_values.append(
            (pred_counts.get(float(cls), 0) / total_pred) * 100)

    distribution_ax.bar(dist_x - dist_width/2, act_pct_values,
                        dist_width, label='Actual %')
    distribution_ax.bar(dist_x + dist_width/2, pred_pct_values,
                        dist_width, label='Predicted %')
    distribution_ax.set_title('Class Distribution')
    distribution_ax.set_ylim(
        0, max(max(act_pct_values), max(pred_pct_values)) * 1.2)
    distribution_ax.set_xticks(dist_x)
    distribution_ax.set_xticklabels([f"{i}" for i in range(len(classes))])
    distribution_ax.legend(fontsize='small')


def _analyze_key_recall_insights(metrics):
    """
    Analyze and display key insights about recall performance.

    Args:
        metrics: Dictionary with extracted metrics
    """
    # Create a summary table of key findings
    insights_table = create_table("Key Recall Insights", [
                                  "Class", "Metric", "Value", "Insight"])

    # Find the class with the highest recall
    best_recall_idx = np.argmax(metrics['recalls'])
    add_table_row(insights_table, [
        metrics['classes'][best_recall_idx],
        "Recall",
        f"{metrics['recalls'][best_recall_idx]:.4f}",
        f"Best recall achieved for this class"
    ])

    # Find the class with the lowest recall
    worst_recall_idx = np.argmin(metrics['recalls'])
    add_table_row(insights_table, [
        metrics['classes'][worst_recall_idx],
        "Recall",
        f"{metrics['recalls'][worst_recall_idx]:.4f}",
        f"Lowest recall for this class"
    ])

    # Find extreme precision values
    high_prec_idx = np.argmax(metrics['precisions'])
    if metrics['precisions'][high_prec_idx] > 0.9 and metrics['recalls'][high_prec_idx] < 0.4:
        add_table_row(insights_table, [
            metrics['classes'][high_prec_idx],
            "Prec/Recall",
            f"{metrics['precisions'][high_prec_idx]:.4f}/{metrics['recalls'][high_prec_idx]:.4f}",
            f"Warning: Very high precision but low recall (conservative predictions)"
        ])

    # Check for balanced predictions
    balanced_idx = np.argmin(
        np.abs(np.array(metrics['precisions']) - np.array(metrics['recalls'])))
    add_table_row(insights_table, [
        metrics['classes'][balanced_idx],
        "Prec/Recall",
        f"{metrics['precisions'][balanced_idx]:.4f}/{metrics['recalls'][balanced_idx]:.4f}",
        f"Most balanced precision-recall for this class"
    ])

    display_table(insights_table)


def _generate_recommendations(metrics, has_skew):
    """
    Generate and display recommendations based on analysis.

    Args:
        metrics: Dictionary with extracted metrics
        has_skew: Boolean indicating if there's significant skew
    """
    # Show recommended next steps based on analysis
    next_steps_table = create_table("Recommended Next Steps", [
                                    "Observation", "Action"])

    # Check for imbalanced distribution
    if has_skew:
        add_table_row(next_steps_table, [
            "Prediction distribution is skewed",
            "Reduce class weight scale or try a different balancing method"
        ])

    # Check for very high precision / low recall
    if max(metrics['precisions']) > 0.9 and min(metrics['recalls']) < 0.3:
        add_table_row(next_steps_table, [
            "Extreme precision-recall trade-off",
            "Adjust decision threshold or try direct calibration of probabilities"
        ])

    # Check for poor F1 scores
    if min(metrics['f1_scores']) < 0.4:
        worst_f1_idx = np.argmin(metrics['f1_scores'])
        add_table_row(next_steps_table, [
            f"Low F1 score for {metrics['classes'][worst_f1_idx]}",
            f"Focus on improving features predictive of this class"
        ])

    display_table(next_steps_table)
