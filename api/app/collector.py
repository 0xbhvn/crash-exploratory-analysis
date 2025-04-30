#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collector script for crash game data.

This script runs in a separate container and is responsible for:
1. Fetching incremental game data
2. Processing game data into streaks
3. Making predictions for 10× streaks
4. Sending notifications for high-confidence predictions
"""

import os
import time
import json
import logging
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL", "http://api:8000")
CSV_FILE = os.getenv("CSV_FILE", "games_live.csv")
MULTIPLIER_THRESHOLD = float(os.getenv("MULTIPLIER_THRESHOLD", "10.0"))
PREDICTION_CONFIDENCE_THRESHOLD = float(
    os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.75"))
SLEEP_SECONDS = int(os.getenv("COLLECTOR_SLEEP_SECONDS", "30"))

# Optional notification endpoints
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def fetch_incremental_data(output_file: str = CSV_FILE, last_game_id: Optional[int] = None) -> bool:
    """
    Fetch incremental game data using the API.
    """
    try:
        # Get the last game ID from the API if not provided
        if last_game_id is None:
            response = requests.get(f"{API_URL}/games/last_id", timeout=10)
            if response.status_code == 200:
                last_game_id = response.json()
                logger.info(f"Last game ID from API: {last_game_id}")
            else:
                logger.warning(
                    f"Could not get last game ID from API: {response.status_code} {response.text}")

        # Import fetch function from main project
        from fetch_data import fetch_incremental_data as fetch_data
        result = fetch_data(output_file, last_game_id, MULTIPLIER_THRESHOLD)
        return result
    except Exception as e:
        logger.error(f"Error fetching incremental data: {str(e)}")
        return False


def process_streaks(csv_file: str = CSV_FILE) -> pd.DataFrame:
    """
    Process game data to identify streaks.
    """
    try:
        # Load game data
        df_games = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df_games)} games from {csv_file}")

        # Import processing function from main project
        from data_processing import extract_streaks_and_multipliers

        # Extract streaks
        streaks_df = extract_streaks_and_multipliers(
            df_games, MULTIPLIER_THRESHOLD)
        logger.info(f"Extracted {len(streaks_df)} streaks")

        return streaks_df
    except Exception as e:
        logger.error(f"Error processing streaks: {str(e)}")
        return pd.DataFrame()


def make_prediction(streaks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Make a prediction for the next streak using the API.
    """
    try:
        # Get the last 50 streaks
        recent_streaks = streaks_df.tail(50).to_dict('records')

        # Send prediction request to API
        response = requests.post(
            f"{API_URL}/predictions/",
            json={"recent_streaks": recent_streaks, "lookback": 50},
            timeout=15
        )

        if response.status_code == 200:
            prediction = response.json()
            logger.info(
                f"Prediction made with confidence: {prediction['confidence']:.4f}")
            return prediction
        else:
            logger.error(
                f"Error making prediction: {response.status_code} {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        return {}


def send_slack_notification(prediction: Dict[str, Any]) -> bool:
    """
    Send a notification to Slack for high-confidence predictions.
    """
    if not SLACK_WEBHOOK:
        return False

    try:
        # Create message
        message = {
            "text": "🎮 Crash Game Streak Prediction 🎮",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎮 Crash Game Streak Prediction Alert 🎮",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Next Streak:* #{prediction['next_streak_number']} (after game #{prediction['starts_after_game_id']})\n"
                                f"*Prediction:* {prediction['prediction_desc']}\n"
                                f"*Confidence:* {prediction['confidence']:.2%}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Short (1-3):* {prediction['prob_class_0']:.2%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Medium-Short (4-7):* {prediction['prob_class_1']:.2%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Medium-Long (8-14):* {prediction['prob_class_2']:.2%}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Long (>14):* {prediction['prob_class_3']:.2%}"
                        }
                    ]
                }
            ]
        }

        # Send request
        response = requests.post(SLACK_WEBHOOK, json=message, timeout=5)
        if response.status_code == 200:
            logger.info("Slack notification sent successfully")
            return True
        else:
            logger.error(
                f"Error sending Slack notification: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Slack notification: {str(e)}")
        return False


def send_telegram_notification(prediction: Dict[str, Any]) -> bool:
    """
    Send a notification to Telegram for high-confidence predictions.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    try:
        # Create message
        message = (
            f"🎮 *Crash Game Streak Prediction Alert* 🎮\n\n"
            f"*Next Streak:* #{prediction['next_streak_number']} (after game #{prediction['starts_after_game_id']})\n"
            f"*Prediction:* {prediction['prediction_desc']}\n"
            f"*Confidence:* {prediction['confidence']:.2%}\n\n"
            f"*Probabilities:*\n"
            f"- Short (1-3): {prediction['prob_class_0']:.2%}\n"
            f"- Medium-Short (4-7): {prediction['prob_class_1']:.2%}\n"
            f"- Medium-Long (8-14): {prediction['prob_class_2']:.2%}\n"
            f"- Long (>14): {prediction['prob_class_3']:.2%}"
        )

        # Send request
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }

        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            logger.info("Telegram notification sent successfully")
            return True
        else:
            logger.error(
                f"Error sending Telegram notification: {response.status_code} {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Telegram notification: {str(e)}")
        return False


def save_prediction_to_file(prediction: Dict[str, Any], filename: str = "predicted_next_streak.json"):
    """
    Save prediction to a local JSON file.
    """
    try:
        with open(filename, 'w') as f:
            json.dump(prediction, f, indent=2)
        logger.info(f"Prediction saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving prediction to file: {str(e)}")


def main():
    """
    Main function for the collector.
    """
    logger.info("Starting collector service...")

    # Ensure output directory exists
    Path("output").mkdir(exist_ok=True)

    while True:
        try:
            # 1. Fetch fresh data
            fetch_success = fetch_incremental_data()

            if fetch_success:
                # 2. Process streaks
                streaks_df = process_streaks()

                if not streaks_df.empty:
                    # 3. Check if the last streak has a 10× multiplier
                    last_streak = streaks_df.tail(1).iloc[0]
                    if last_streak["hit_multiplier"] >= MULTIPLIER_THRESHOLD:
                        # 4. Make prediction for the next streak
                        prediction = make_prediction(streaks_df)

                        if prediction and "confidence" in prediction:
                            # 5. Save prediction to file
                            save_prediction_to_file(prediction)

                            # 6. Send notifications for high-confidence predictions
                            if prediction["confidence"] >= PREDICTION_CONFIDENCE_THRESHOLD:
                                logger.info(
                                    f"High confidence prediction detected: {prediction['confidence']:.4f}")
                                send_slack_notification(prediction)
                                send_telegram_notification(prediction)

        except Exception as e:
            logger.error(f"Collector error: {str(e)}")

        # Sleep before next iteration
        logger.info(f"Sleeping for {SLEEP_SECONDS} seconds...")
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
