#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Replay predictor for historical data simulation.

This script replays historical game data through the LivePredictor logic,
simulating what would have happened if the prediction system had been running
in real-time. It ensures temporal integrity by processing games in chronological
order and only using past data for predictions.
"""

from utils.logger_config import (
    console, print_info, print_success, print_warning, print_error, print_panel,
    create_table, display_table, add_table_row, create_stats_table
)
import argparse
import joblib
import json
import os
import time
from collections import deque
from pathlib import Path

import pandas as pd
from rich.progress import Progress

# Import local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from utils

# Import temporal model functions
try:
    from temporal.deploy import load_model_and_predict, setup_prediction_service
except ImportError:
    print_error(
        "Could not import from temporal.deploy - make sure the module exists")
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

            if MODEL_BUNDLE and "percentile_values" in MODEL_BUNDLE:
                print_success(
                    f"Model loaded successfully with percentile values: {MODEL_BUNDLE['percentile_values']}")
                return MODEL_BUNDLE

        # If that fails or doesn't have percentile values, try setup_prediction_service
        print_info("Attempting to load via setup_prediction_service...")
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)

        bundle = setup_prediction_service(
            model_dir=model_dir, model_filename=model_filename)

        if bundle and bundle.get("percentile_values"):
            MODEL_BUNDLE = bundle
            print_success(
                f"Model loaded via service with percentile values: {MODEL_BUNDLE['percentile_values']}")
            return MODEL_BUNDLE

        # If we've gotten here without setting MODEL_BUNDLE, create a minimal one
        if not MODEL_BUNDLE:
            print_warning(
                f"Could not load model properly. Creating minimal bundle with default percentiles.")
            MODEL_BUNDLE = {
                "path": str(model_path),
                "percentile_values": DEFAULT_PERCENTILES
            }

        return MODEL_BUNDLE
    except Exception as e:
        print_error(f"Error loading model bundle: {str(e)}")
        MODEL_BUNDLE = {
            "path": str(model_path),
            "percentile_values": DEFAULT_PERCENTILES
        }
        print_warning(
            f"Using fallback model path with default percentiles: {DEFAULT_PERCENTILES}")
        return MODEL_BUNDLE


class ReplayPredictor:
    """Replay version of LivePredictor for historical data simulation."""

    def __init__(self, model_path, window=50, output_file="replay_predictions.log"):
        """Initialize the replay predictor."""
        global MODEL_BUNDLE

        print_info(f"Setting up prediction service with model: {model_path}")
        MODEL_BUNDLE = load_model_bundle(model_path)

        # Extract and display percentile values for transparency
        self.percentile_values = MODEL_BUNDLE.get(
            "percentile_values", DEFAULT_PERCENTILES)
        print_info(
            f"Using percentile values for clustering: {self.percentile_values}")

        # Create percentile-based label mapping for better output
        p25, p50, p75 = self.percentile_values
        self.cluster_labels = {
            0: f"short (≤{p25})",
            1: f"medium_short ({p25+0.1}-{p50})",
            2: f"medium_long ({p50+0.1}-{p75})",
            3: f"long (>{p75})"
        }

        # Display cluster boundaries
        clusters_table = create_table(
            "Cluster Boundaries", ["Cluster", "Label", "Range"])
        add_table_row(clusters_table, ["0", "Short", f"≤ {p25}"])
        add_table_row(clusters_table, [
                      "1", "Medium-Short", f"{p25+0.1} - {p50}"])
        add_table_row(clusters_table, [
                      "2", "Medium-Long", f"{p50+0.1} - {p75}"])
        add_table_row(clusters_table, ["3", "Long", f"> {p75}"])
        display_table(clusters_table)

        self.lookback = window
        self.min_lookback = MIN_LOOKBACK  # Configurable minimum lookback
        self.history = deque(maxlen=KEEP_HISTORY)
        self.current = []  # bust multipliers of ongoing streak
        self.pending = None  # last prediction awaiting truth
        self.output_file = output_file
        self.stats = {
            "total_games": 0,
            "total_streaks": 0,
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0.0
        }

        # Initialize output file
        with open(self.output_file, "w") as f:
            f.write("# Replay Predictions Log\n")
            f.write(f"# Percentile values: {self.percentile_values}\n")
            f.write(f"# Cluster mapping: {json.dumps(self.cluster_labels)}\n")

    def handle_bust(self, game_id: int, bust: float, timestamp=None):
        """Process a single game bust value."""
        self.stats["total_games"] += 1
        self.current.append(
            {"Game ID": game_id, "Bust": bust, "timestamp": timestamp or time.time()})

        if bust >= 10.0:  # 10× hit --> streak closes
            streak_df = self._finalise_current_streak()
            self._score_previous_if_any(streak_df)
            self._predict_next(streak_df)

    def _finalise_current_streak(self) -> pd.DataFrame:
        """Finalize the current streak and add to history."""
        streak_number = len(self.history) + 1
        df = pd.DataFrame(self.current)

        # Calculate streak stats similar to streak_processor.py
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
        self.current = []  # reset for next streak

        return pd.DataFrame([streak])

    def _score_previous_if_any(self, truth_df):
        """Score the previous prediction if one exists."""
        if not self.pending:
            return

        truth_length = truth_df.iloc[0]["streak_length"]
        truth_cluster = self._cluster(truth_length)
        correct = truth_cluster == self.pending["predicted_cluster"]

        self.stats["total_predictions"] += 1
        if correct:
            self.stats["correct_predictions"] += 1

        self.stats["accuracy"] = (
            self.stats["correct_predictions"] / self.stats["total_predictions"]
            if self.stats["total_predictions"] > 0
            else 0.0
        )

        self.pending.update({
            "actual_cluster": truth_cluster,
            "actual_streak_length": truth_length,
            "correct": correct,
            "resolved_at": time.time()
        })

        self._log("truth", self.pending)
        self.pending = None

    def _predict_next(self, streak_df):
        """Predict the next streak characteristics."""
        if len(self.history) < self.min_lookback:
            print_warning(
                f"Not enough history yet: {len(self.history)}/{self.min_lookback} streaks minimum required")
            return

        # Determine best lookback window based on available history
        actual_lookback = min(self.lookback, len(self.history))
        if actual_lookback < self.lookback:
            print_info(
                f"Using reduced lookback window: {actual_lookback}/{self.lookback} (limited by available history)")

        # Build a DataFrame of most-recent streaks
        hist_df = pd.DataFrame(list(self.history))[-actual_lookback:]

        try:
            # Add more debug information
            print_info(
                f"Making prediction with {len(hist_df)} historical streaks")

            # Make prediction using the model's path
            model_path = MODEL_BUNDLE.get(
                'path', str(Path("output/temporal_model.pkl")))

            prediction = load_model_and_predict(model_path, hist_df)

            if prediction:
                # Include percentile information in the prediction log for analysis
                prediction["model_percentiles"] = self.percentile_values
                prediction["cluster_desc"] = self.cluster_labels.get(
                    prediction.get("predicted_cluster", 0), "unknown")

                # Log and store the prediction
                self._log("pred", prediction)
                self.pending = prediction
            else:
                print_error("Model returned empty prediction")

        except Exception as e:
            print_error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()

    def _cluster(self, length: int) -> int:
        """Map streak length to cluster using model's saved percentiles."""
        if length <= self.percentile_values[0]:
            return 0
        if length <= self.percentile_values[1]:
            return 1
        if length <= self.percentile_values[2]:
            return 2
        return 3

    def _log(self, kind, obj):
        """Log prediction/validation to file and console."""
        # Remove non-serializable entries if any
        log_obj = {k: v for k, v in obj.items() if isinstance(
            v, (str, int, float, bool, list, dict)) or v is None}

        # Add timestamp and kind
        log_entry = {"t": time.time(), "kind": kind, **log_obj}

        # Add percentile values for validation
        if kind == "truth" and "actual_streak_length" in log_obj:
            # Reclassify the actual streak length using current percentiles
            length = log_obj.get("actual_streak_length")
            cluster = self._cluster(length)
            log_entry["actual_cluster"] = cluster
            log_entry["model_percentiles"] = self.percentile_values
            log_entry["correct"] = (
                cluster == log_obj.get("predicted_cluster"))

        # Write to file
        with open(self.output_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Print to console with friendly labels
        if kind == "pred":
            cluster_desc = self.cluster_labels.get(
                log_obj.get("predicted_cluster", 0), "unknown")
            print_success(f"PREDICTION: Streak #{log_obj.get('next_streak_number')} - "
                          f"{cluster_desc} (confidence: {log_obj.get('confidence', 0):.4f})")
        else:
            # Use the reclassified correctness
            correct = log_entry.get("correct", log_obj.get("correct", False))
            correct_str = "✓ CORRECT" if correct else "✗ INCORRECT"

            # Get descriptive labels for predicted and actual clusters
            predicted_label = self.cluster_labels.get(
                log_obj.get("predicted_cluster", 0), "unknown")
            actual_label = self.cluster_labels.get(
                log_entry.get("actual_cluster", 0), "unknown")

            print_info(
                f"VALIDATION: {correct_str} - "
                f"Predicted: {predicted_label}, "
                f"Actual: {actual_label} (length={log_obj.get('actual_streak_length')})")

    def print_summary(self):
        """Print summary statistics after completion."""
        stats_table = {
            "Total Games": self.stats["total_games"],
            "Total Streaks": self.stats["total_streaks"],
            "Total Predictions": self.stats["total_predictions"],
            "Correct Predictions": self.stats["correct_predictions"],
            "Accuracy": f"{self.stats['accuracy'] * 100:.2f}%"
        }
        create_stats_table("Replay Prediction Results", stats_table)


def replay_from_games(predictor, games_df, start_game_id=None, step_by_step=False, auto_continue=False, delay=0.5):
    """Replay game data through the predictor in chronological order."""
    print_panel("Replaying Games CSV", style="blue")

    # Sort by game_id to ensure chronological order
    sorted_games = games_df.sort_values("Game ID")

    # Filter games starting from start_game_id if provided
    if start_game_id:
        sorted_games = sorted_games[sorted_games["Game ID"].astype(
            int) >= start_game_id]
        print_info(
            f"Starting from game ID {start_game_id}, {len(sorted_games)} games to process")

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Replaying games...", total=len(sorted_games))

        for _, row in sorted_games.iterrows():
            # Process each game
            game_id = int(row["Game ID"])
            bust = float(row["Bust"])

            # Handle the bust
            predictor.handle_bust(game_id, bust)

            # If step by step mode is enabled, pause after each game
            if step_by_step:
                print_info(f"Processed game #{game_id} with bust {bust}")
                if auto_continue:
                    import time
                    # Wait a short time instead of requiring input
                    time.sleep(delay)
                else:
                    input("Press Enter to continue to next game...")

            # Update progress
            progress.update(task, advance=1)


def replay_from_streaks(predictor, streaks_df, games_df, step_by_step=False, auto_continue=False, delay=0.5):
    """Replay streak data through the predictor, fetching game data as needed."""
    print_panel("Replaying from Streaks CSV", style="blue")

    with Progress() as progress:
        task = progress.add_task(
            "[cyan]Replaying streaks...", total=len(streaks_df))

        # Sort by streak_number to ensure chronological order
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

            # Get all games in this streak from games_df
            streak_games = games_df[
                (games_df["Game ID"].astype(int) >= start_id) &
                (games_df["Game ID"].astype(int) <= end_id)
            ].sort_values("Game ID")

            if len(streak_games) == 0:
                print_warning(
                    f"No games found for streak {streak['streak_number']} (IDs {start_id}-{end_id})")
                continue

            # Process each game in the streak
            for idx, game in enumerate(streak_games.iterrows()):
                _, row = game
                game_id = int(row["Game ID"])
                bust = float(row["Bust"])
                predictor.handle_bust(game_id, bust)

                # If step by step mode and not the last game in streak, pause between games
                if step_by_step and idx < len(streak_games) - 1:
                    print_info(f"Processed game #{game_id} with bust {bust}")
                    if auto_continue:
                        import time
                        time.sleep(delay)
                    else:
                        input("Press Enter for next game in this streak...")

            # Update progress
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
    parser.add_argument("--output", default="replay_predictions.log", type=str,
                        help="Output file for predictions log")
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

    # Update global MIN_LOOKBACK
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
        f"Output: {args.output}",
        title="Configuration",
        style="green"
    )

    # Initialize the replay predictor
    predictor = ReplayPredictor(
        model_path=Path(args.model),
        window=args.window,
        output_file=args.output
    )

    # Load games data
    try:
        games_df = pd.read_csv(args.games)
        print_info(f"Loaded {len(games_df)} games from {args.games}")
    except Exception as e:
        print_error(f"Error loading games CSV: {str(e)}")
        return

    # Process data based on mode
    if args.use_streaks:
        try:
            streaks_df = pd.read_csv(args.streaks)
            print_info(f"Loaded {len(streaks_df)} streaks from {args.streaks}")

            # If starting from a specific game ID, filter streaks accordingly
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

    # Print summary statistics
    predictor.print_summary()

    print_success(f"Replay completed. Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
