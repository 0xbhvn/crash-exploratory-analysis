#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for Crash Game 10× Streak Analysis.

This module handles plotting and visualization of data.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

logger = logging.getLogger(__name__)


def plot_streaks(streak_lengths: List[int], percentiles: Dict[str, float], output_dir: str) -> None:
    """
    Generate and save plots of streak length distributions.

    Args:
        streak_lengths: List of streak lengths
        percentiles: Dictionary of percentiles
        output_dir: Directory to save plots
    """
    logger.info("Generating streak length distribution plots")

    # Basic histogram
    plt.figure(figsize=(10, 6))
    bins = range(1, max(streak_lengths) + 2)
    plt.hist(streak_lengths, bins=bins, edgecolor="black", alpha=0.8)
    plt.title("Distribution of streak lengths BEFORE a ≥10× bust")
    plt.xlabel("Streak length (# consecutive <10× busts)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "streak_histogram.png"))
    plt.close()

    # Histogram with percentiles
    plt.figure(figsize=(12, 6))
    plt.hist(streak_lengths, bins=bins, edgecolor='black', alpha=0.7)
    for label, value in percentiles.items():
        plt.axvline(value, color='red', linestyle='--')
        plt.text(value + 0.3, plt.ylim()[1] * 0.9, f"{label}: {value:.1f}",
                 rotation=90, color='red')

    plt.title("Distribution of streak lengths (with percentiles)")
    plt.xlabel("Streak length")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "streak_percentiles.png"))
    plt.close()

    logger.info("Saved streak distribution plots")


def plot_feature_importance(model, feature_cols: List[str], output_dir: str) -> None:
    """
    Plot feature importance from the trained model.

    Args:
        model: Trained XGBoost model
        feature_cols: List of feature column names
        output_dir: Directory to save plots
    """
    logger.info("Generating feature importance plot")

    # Get feature importance
    importance = model.get_score(importance_type='gain')

    # Convert to DataFrame for easier plotting
    import pandas as pd
    imp_df = pd.DataFrame({
        'Feature': list(importance.keys()),
        'Importance': list(importance.values())
    }).sort_values('Importance', ascending=False)

    # Limit to top 20 features for readability
    if len(imp_df) > 20:
        imp_df = imp_df.head(20)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['Feature'], imp_df['Importance'])
    plt.xlabel('Gain Importance')
    plt.title('Feature Importance (Gain)')
    plt.gca().invert_yaxis()  # Display with most important at the top
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"))
    plt.close()

    logger.info("Saved feature importance plot")


def plot_calibration_curve(y_test, probs_test, output_dir: str) -> None:
    """
    Plot calibration curve for model evaluation.

    Args:
        y_test: Test labels
        probs_test: Predicted probabilities
        output_dir: Directory to save plots
    """
    logger.info("Generating calibration curve")

    from sklearn.calibration import calibration_curve

    # Use the first class (short streaks) for binary calibration
    prob_true, prob_pred = calibration_curve(
        y_test == 0, probs_test[:, 0], n_bins=10)

    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')

    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve (Short streak class)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
    plt.close()

    logger.info("Saved calibration curve plot")
