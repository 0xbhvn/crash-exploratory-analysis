#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replay predictor for historical data simulation.

This script replays historical game data through the LivePredictor logic,
simulating what would have happened if the prediction system had been running
in real-time. It ensures temporal integrity by processing games in chronological
order and only using past data for predictions.
"""

# --- Suppress Pandas PerformanceWarning ---
import sys
import xgboost as xgb
import numpy as np
from rich.progress import Progress
from pathlib import Path
from collections import deque
import csv
import time
import os
import json
import joblib
import argparse
from utils.logger_config import (
    console, print_info, print_success, print_warning, print_error, print_panel,
    create_table, display_table, add_table_row, create_stats_table
)
import warnings
import pandas as pd
try:
    # pandas >= 1.5.0
    from pandas.errors import PerformanceWarning
except ImportError:
    # pandas < 1.5.0
    from pandas.core.common import PerformanceWarning
warnings.filterwarnings('ignore', category=PerformanceWarning)
# --- End Warning Suppression ---


# Import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import temporal model functions and feature creation
try:
    from temporal.deploy import load_model_and_predict, setup_prediction_service
    from temporal.features import create_temporal_features
except ImportError:
    print_error(
        "Could not import from temporal.deploy or temporal.features - make sure the modules exist")
    sys.exit(1)

KEEP_HISTORY = 400  # streaks to keep for feature window
MODEL_BUNDLE = None
MIN_LOOKBACK = 10   # Minimum number of streaks needed for prediction
# Default percentiles if model doesn't have them
DEFAULT_PERCENTILES = [3, 7, 14]


def load_model_bundle(model_path):
    """Load the model bundle and extract percentile values."""
    global MODEL_BUNDLE

    try:
        print_info(f"Loading model bundle directly from {model_path}")
        # First try direct loading with joblib
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                MODEL_BUNDLE = joblib.load(f)

            # Check if essential keys are present
            if MODEL_BUNDLE and 'model' in MODEL_BUNDLE and 'scaler' in MODEL_BUNDLE and 'feature_cols' in MODEL_BUNDLE:
                # Attempt to get percentile values, use default if not found
                percentiles = MODEL_BUNDLE.get(
                    'percentile_values', DEFAULT_PERCENTILES)
                MODEL_BUNDLE['percentile_values'] = percentiles
                print_success(
                    f"Model loaded successfully with percentile values: {percentiles}")
                return MODEL_BUNDLE

        # If direct load failed or bundle was incomplete, try setup_prediction_service
        print_info(
            "Direct load failed or incomplete. Attempting via setup_prediction_service...")
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)

        bundle = setup_prediction_service(
            model_dir=model_dir, model_filename=model_filename)

        if bundle and bundle.get('model') and bundle.get('scaler') and bundle.get('feature_cols'):
            MODEL_BUNDLE = bundle
            # Ensure percentile values are present, use default if not
            percentiles = MODEL_BUNDLE.get(
                'percentile_values', DEFAULT_PERCENTILES)
            MODEL_BUNDLE['percentile_values'] = percentiles
            print_success(
                f"Model loaded via service with percentile values: {percentiles}")
            return MODEL_BUNDLE

        # Fallback if both methods fail
        print_error("Failed to load a valid model bundle from either method.")
        raise ValueError("Could not load a valid model bundle.")

    except Exception as e:
        print_error(f"Error loading model bundle: {str(e)}")
        raise  # Re-raise the exception after logging


class ReplayPredictor:
    """Replay version of LivePredictor for historical data simulation."""

    def __init__(self, model_path, window=50, output_file="replay_predictions.csv"):
        """Initialize the replay predictor."""
        global MODEL_BUNDLE

        print_info(f"Setting up prediction service with model: {model_path}")
        # Ensure MODEL_BUNDLE is loaded correctly
        if not MODEL_BUNDLE or MODEL_BUNDLE.get('path') != str(model_path):
            MODEL_BUNDLE = load_model_bundle(model_path)

        # Extract percentile values AFTER ensuring bundle is loaded
        self.percentile_values = MODEL_BUNDLE.get(
            'percentile_values', DEFAULT_PERCENTILES)
        print_info(
            f"Using percentile values for clustering: {self.percentile_values}")

        # Create percentile-based label mapping
        p25, p50, p75 = self.percentile_values
        self.cluster_labels = {
            0: f"short (≤{p25})",
            # Ensure correct float formatting
            1: f"medium_short ({p25+0.1:.1f}-{p50})",
            2: f"medium_long ({p50+0.1:.1f}-{p75})",
            3: f"long (>{p75})"
        }

        clusters_table = create_table(
            "Cluster Boundaries", ["Cluster", "Label", "Range"])
        add_table_row(clusters_table, ["0", "Short", f"≤ {p25}"])
        add_table_row(clusters_table, [
                      "1", "Medium-Short", f"{p25+0.1:.1f} - {p50}"])
        add_table_row(clusters_table, [
                      "2", "Medium-Long", f"{p50+0.1:.1f} - {p75}"])
        add_table_row(clusters_table, ["3", "Long", f"> {p75}"])
        display_table(clusters_table)

        self.lookback = window
        self.min_lookback = MIN_LOOKBACK
        self.history = deque(maxlen=KEEP_HISTORY)
        self.current = []
        self.pending = None
        self.output_file = output_file
        self.stats = {
            "total_games": 0,
            "total_streaks": 0,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0
        }

        # --- Initialize CSV file ---
        self.csv_headers = [
            'timestamp',
            'predicted_for_streak',
            'predicted_cluster',
            'prediction_desc',
            'confidence',
            'prob_class_0',
            'prob_class_1',
            'prob_class_2',
            'prob_class_3',
            'actual_streak_length',
            'actual_cluster',
            'actual_desc',
            'correct'
        ]
        try:
            with open(self.output_file, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)
            print_info(f"Initialized CSV log at {self.output_file}")
        except Exception as e:
            print_error(f"Failed to initialize CSV log file: {e}")
            # Decide whether to raise or continue without logging
            raise
        # --- End CSV Initialization ---

    def handle_bust(self, game_id: int, bust: float, timestamp=None):
        """Process a single game bust value."""
        self.stats["total_games"] += 1
        self.current.append(
            {"Game ID": game_id, "Bust": bust, "timestamp": timestamp or time.time()})

        if bust >= 10.0:
            streak_df = self._finalise_current_streak()
            self._score_previous_if_any(streak_df)
            # Pass the just completed streak for context if needed
            self._predict_next()

    def _finalise_current_streak(self) -> pd.DataFrame:
        """Finalize the current streak and add to history."""
        streak_number = len(self.history) + 1
        df = pd.DataFrame(self.current)
        streak = {
            "streak_number": streak_number,
            "start_game_id": int(df["Game ID"].iloc[0]),
            "end_game_id": int(df["Game ID"].iloc[-1]),
            "streak_length": len(df),
            "hit_multiplier": float(df["Bust"].iloc[-1]),
            "mean_multiplier": float(df["Bust"].mean()),
            "max_multiplier": float(df["Bust"].max()),
            "min_multiplier": float(df["Bust"].min()),
            "std_multiplier": float(df["Bust"].std()) if len(df) > 1 else 0.0,
            "temporal_idx": streak_number - 1,
        }
        self.history.append(streak)
        self.stats["total_streaks"] += 1
        self.current = []
        return pd.DataFrame([streak])

    def _score_previous_if_any(self, truth_df):
        """Score the previous prediction and log to CSV if one exists."""
        if not self.pending:
            return

        try:
            # --- Calculate Truth ---
            truth_length = truth_df.iloc[0]["streak_length"]
            truth_cluster = self._cluster(truth_length)
            correct = truth_cluster == self.pending["predicted_cluster"]
            actual_desc = self.cluster_labels.get(truth_cluster, "unknown")

            # --- Update Stats ---
            self.stats["total_predictions"] += 1
            if correct:
                self.stats["correct_predictions"] += 1

            # --- Prepare CSV Data ---
            csv_data = {
                # Use prediction time if available
                'timestamp': self.pending.get('prediction_timestamp', time.time()),
                'predicted_for_streak': self.pending['next_streak_number'],
                'predicted_cluster': self.pending['predicted_cluster'],
                'prediction_desc': self.pending['cluster_desc'],
                # Format confidence
                'confidence': f"{self.pending['confidence']:.4f}",
                'prob_class_0': f"{self.pending.get('prob_class_0', 0):.4f}",
                'prob_class_1': f"{self.pending.get('prob_class_1', 0):.4f}",
                'prob_class_2': f"{self.pending.get('prob_class_2', 0):.4f}",
                'prob_class_3': f"{self.pending.get('prob_class_3', 0):.4f}",
                'actual_streak_length': truth_length,
                'actual_cluster': truth_cluster,
                'actual_desc': actual_desc,
                'correct': correct
            }

            # --- Write to CSV ---
            with open(self.output_file, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(csv_data)

        except Exception as e:
            print_error(
                f"Error during scoring/logging for streak {self.pending.get('next_streak_number', 'N/A')}: {e}")

        finally:
            self.pending = None  # Ensure pending is cleared

    def _predict_next(self):
        """Predict the next streak characteristics."""
        if len(self.history) < self.min_lookback:
            # Suppress repetitive warning
            return

        # Determine best lookback window based on available history
        actual_lookback = min(self.lookback, len(self.history))
        # Log if lookback is reduced? Only if significantly different?
        # if actual_lookback < self.lookback:
        #     print_info(
        #         f"Using reduced lookback window: {actual_lookback}/{self.lookback} (limited by available history)")

        # Build a DataFrame of most-recent streaks from history
        hist_df = pd.DataFrame(list(self.history))[-actual_lookback:]

        try:
            # Add more debug information
            # print_info(
            #     f"Making prediction based on {len(hist_df)} historical streaks (numbers {hist_df['streak_number'].min()}-{hist_df['streak_number'].max()})")

            # === Feature Generation Step ===
            model = MODEL_BUNDLE['model']  # Calibrator
            scaler = MODEL_BUNDLE['scaler']
            feature_cols = MODEL_BUNDLE['feature_cols']

            # Generate the required temporal features from the historical streak data
            # print_info(f"Generating temporal features for prediction...")
            features_for_prediction, _, _ = create_temporal_features(
                hist_df, lookback_window=actual_lookback)

            if features_for_prediction.empty:
                print_error(
                    "Feature generation returned empty DataFrame. Cannot predict.")
                return

            last_features_row = features_for_prediction.iloc[[-1]]

            # Handle potential missing columns
            missing_cols = [
                col for col in feature_cols if col not in last_features_row.columns]
            if missing_cols:
                print_warning(
                    f"Generated features missing columns: {missing_cols}. Adding defaults (0). This might indicate an issue!")
                for col in missing_cols:
                    # Use .loc to avoid SettingWithCopyWarning
                    last_features_row.loc[:, col] = 0

            # Ensure columns are in the correct order
            X_predict = last_features_row[feature_cols]
            # === End Feature Generation ===

            # Scale the single row of generated features
            X_scaled = scaler.transform(X_predict)

            # Use the main model (Calibrator) directly
            # print_info("Using calibrated model for replay prediction probabilities.")
            y_pred_proba = model.predict_proba(X_scaled)

            # Prediction is for a single next event, so result should be shape (1, n_classes)
            if y_pred_proba.shape == (1, len(self.cluster_labels)):
                last_proba = y_pred_proba[0]
                y_pred = np.argmax(last_proba)
                confidence = np.max(last_proba)
                prediction_valid = True
            else:
                print_error(
                    f"Unexpected probability shape: {y_pred_proba.shape}. Expected (1, {len(self.cluster_labels)}). Cannot extract prediction.")
                prediction_valid = False
                y_pred = -1
                confidence = 0.0
                last_proba = np.zeros(len(self.cluster_labels))

            # Store prediction details if valid
            if prediction_valid:
                # Store prediction details in self.pending
                self.pending = {
                    'prediction_timestamp': time.time(),  # Record prediction time
                    'next_streak_number': int(hist_df['streak_number'].iloc[-1] + 1),
                    'predicted_cluster': int(y_pred),
                    'confidence': float(confidence),
                    'cluster_desc': self.cluster_labels.get(int(y_pred), "unknown")
                    # Store individual probabilities
                }
                for i in range(len(last_proba)):
                    self.pending[f'prob_class_{i}'] = float(last_proba[i])

                # Optional: Minimal console log for prediction made
                print_success(
                    f"PREDICTION MADE: Streak #{self.pending['next_streak_number']} - {self.pending['cluster_desc']} (conf: {self.pending['confidence']:.2f})")

            else:
                print_error(
                    "Prediction deemed invalid due to probability shape issues.")
                self.pending = None  # Ensure pending is clear if prediction fails

        except KeyError as e:
            print_error(
                f"Prediction error: Missing expected column - {str(e)}. This might be a feature mismatch.")
            import traceback
            traceback.print_exc()
            self.pending = None  # Clear pending on error
        except Exception as e:
            print_error(f"General prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.pending = None  # Clear pending on error

    def _cluster(self, length: int) -> int:
        """Map streak length to cluster using model's saved percentiles."""
        p25, p50, p75 = self.percentile_values
        if length <= p25:
            return 0
        if length <= p50:
            return 1
        if length <= p75:
            return 2
        return 3

    def print_summary(self):
        """Print summary statistics after completion."""
        # Add check for zero predictions to avoid division by zero
        if self.stats["total_predictions"] > 0:
            accuracy_val = self.stats['correct_predictions'] / \
                self.stats['total_predictions']
        else:
            accuracy_val = 0.0

        stats_table_data = {
            "Total Games": self.stats["total_games"],
            "Total Streaks": self.stats["total_streaks"],
            "Total Predictions Made": self.stats["total_predictions"],
            "Correct Predictions": self.stats["correct_predictions"],
            "Accuracy": f"{accuracy_val * 100:.2f}%"  # Use calculated accuracy
        }
        # Use the existing create_stats_table function
        create_stats_table("Replay Prediction Results", stats_table_data)


def replay_from_games(predictor, games_df, start_game_id=None, step_by_step=False, auto_continue=False, delay=0.5):
    """Replay game data through the predictor in chronological order."""
    print_panel("Replaying Games CSV", style="blue")
    sorted_games = games_df.sort_values("Game ID")
    if start_game_id:
        sorted_games = sorted_games[sorted_games["Game ID"].astype(
            int) >= start_game_id]
        print_info(
            f"Starting from game ID {start_game_id}, {len(sorted_games)} games to process")
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Replaying games...", total=len(sorted_games))
        for _, row in sorted_games.iterrows():
            game_id = int(row["Game ID"])
            bust = float(row["Bust"])
            predictor.handle_bust(game_id, bust)
            if step_by_step:
                print_info(f"Processed game #{game_id} with bust {bust}")
                if auto_continue:
                    import time
                    time.sleep(delay)
                else:
                    input("Press Enter to continue to next game...")
            progress.update(task, advance=1)


def replay_from_streaks(predictor, streaks_df, games_df, step_by_step=False, auto_continue=False, delay=0.5):
    """Replay streak data through the predictor, fetching game data as needed."""
    print_panel("Replaying from Streaks CSV", style="blue")
    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Replaying streaks...", total=len(streaks_df))
        sorted_streaks = streaks_df.sort_values("streak_number")
        for _, streak in sorted_streaks.iterrows():
            start_id = int(streak["start_game_id"])
            end_id = int(streak["end_game_id"])
            if step_by_step:
                print_info(
                    f"Processing streak #{streak['streak_number']} (games {start_id}-{end_id})")
                if auto_continue:
                    import time
                    time.sleep(delay)
                else:
                    input("Press Enter to process this streak...")
            streak_games = games_df[
                (games_df["Game ID"].astype(int) >= start_id) &
                (games_df["Game ID"].astype(int) <= end_id)
            ].sort_values("Game ID")
            if len(streak_games) == 0:
                print_warning(
                    f"No games found for streak {streak['streak_number']} (IDs {start_id}-{end_id})")
                continue
            for idx, game in enumerate(streak_games.iterrows()):
                _, row = game
                game_id = int(row["Game ID"])
                bust = float(row["Bust"])
                predictor.handle_bust(game_id, bust)
                if step_by_step and idx < len(streak_games) - 1:
                    print_info(f"Processed game #{game_id} with bust {bust}")
                    if auto_continue:
                        import time
                        time.sleep(delay)
                    else:
                        input("Press Enter for next game in this streak...")
            progress.update(task, advance=1)
            if step_by_step:
                print_info(f"Completed streak #{streak['streak_number']}")
                if auto_continue:
                    import time
                    time.sleep(delay)
                else:
                    input("Press Enter to continue to next streak...")


def main():
    """Main function to run the replay predictor."""
    parser = argparse.ArgumentParser(
        description="Replay historical data through prediction model")
    parser.add_argument("--model", default="output/temporal_model.pkl", type=str,
                        help="Path to the model pickle file")
    parser.add_argument("--games", default="games.csv", type=str,
                        help="Path to games CSV file")
    parser.add_argument("--streaks", default="output/streak_lengths.csv", type=str,
                        help="Path to streaks CSV file (optional)")
    parser.add_argument("--output", default="replay_predictions.csv", type=str,
                        help="Output file for predictions log (CSV format)")
    parser.add_argument("--window", default=50, type=int,
                        help="Ideal lookback window size for predictions")
    parser.add_argument("--min-window", default=10, type=int,
                        help="Minimum lookback window size (lower = earlier predictions)")
    parser.add_argument("--use-streaks", action="store_true",
                        help="Use pre-calculated streaks instead of parsing games")
    parser.add_argument("--start-game-id", type=int, default=None,
                        help="Start processing from this game ID")
    parser.add_argument("--step-by-step", action="store_true",
                        help="Process one game at a time, pausing between each")
    parser.add_argument("--auto-continue", action="store_true",
                        help="Automatically continue after a short delay without requiring Enter key")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay in seconds between steps when using auto-continue (default: 0.5)")

    args = parser.parse_args()

    global MIN_LOOKBACK
    MIN_LOOKBACK = args.min_window

    print_panel(
        f"Replay Predictor\n"
        f"Model: {args.model}\n"
        f"Window: {args.window} (min {args.min_window})\n"
        f"Mode: {'Pre-calculated Streaks' if args.use_streaks else 'Raw Games'}\n"
        f"Start Game ID: {args.start_game_id or 'Beginning'}\n"
        f"Step by Step: {'Yes' if args.step_by_step else 'No'}\n"
        f"Auto Continue: {'Yes' if args.auto_continue else 'No'}\n"
        f"Delay: {args.delay}s\n"
        f"Output: {args.output} (CSV Format)",  # Indicate CSV format
        title="Configuration",
        style="green"
    )

    predictor = ReplayPredictor(
        model_path=Path(args.model),
        window=args.window,
        output_file=args.output
    )

    try:
        games_df = pd.read_csv(args.games)
        print_info(f"Loaded {len(games_df)} games from {args.games}")
    except Exception as e:
        print_error(f"Error loading games CSV: {str(e)}")
        return

    if args.use_streaks:
        try:
            streaks_df = pd.read_csv(args.streaks)
            print_info(f"Loaded {len(streaks_df)} streaks from {args.streaks}")
            if args.start_game_id:
                streaks_df = streaks_df[streaks_df["start_game_id"].astype(
                    int) >= args.start_game_id]
                print_info(
                    f"Filtered to {len(streaks_df)} streaks starting from game ID {args.start_game_id}")
            replay_from_streaks(predictor, streaks_df, games_df,
                                args.step_by_step, args.auto_continue, args.delay)
        except Exception as e:
            print_error(f"Error loading or processing streaks CSV: {str(e)}")
            return
    else:
        replay_from_games(predictor, games_df, args.start_game_id,
                          args.step_by_step, args.auto_continue, args.delay)

    predictor.print_summary()
    print_success(f"Replay completed. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
