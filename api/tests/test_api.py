#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the API.

This script tests the API endpoints.
"""

import os
import sys
import logging
import requests
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
API_URL = os.getenv("API_URL", "http://localhost:8000")
CSV_FILE = os.getenv("CSV_FILE", "../games.csv")
MULTIPLIER_THRESHOLD = float(os.getenv("MULTIPLIER_THRESHOLD", "10.0"))


def test_health_check():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{API_URL}/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        logger.info("Health check test passed")
        return True
    except Exception as e:
        logger.error(f"Health check test failed: {str(e)}")
        return False


def test_prediction():
    """Test prediction endpoint."""
    try:
        # Load games data
        games_path = Path(CSV_FILE)
        if not games_path.exists():
            logger.error(f"Games file not found: {CSV_FILE}")
            return False

        games_df = pd.read_csv(games_path)
        logger.info(f"Loaded {len(games_df)} games from {CSV_FILE}")

        # Process streaks
        from data_processing import extract_streaks_and_multipliers
        streaks_df = extract_streaks_and_multipliers(
            games_df, MULTIPLIER_THRESHOLD)
        logger.info(f"Extracted {len(streaks_df)} streaks")

        if streaks_df.empty:
            logger.error("No streaks found in data")
            return False

        # Use the last 50 streaks
        recent_streaks = streaks_df.tail(50).to_dict('records')

        # Send prediction request
        response = requests.post(
            f"{API_URL}/predictions/",
            json={"recent_streaks": recent_streaks, "lookback": 50},
            timeout=15
        )

        assert response.status_code == 200
        prediction = response.json()
        assert "next_streak_number" in prediction
        assert "confidence" in prediction

        logger.info(
            f"Prediction test passed with confidence: {prediction['confidence']:.4f}")
        logger.info(f"Prediction: {prediction['prediction_desc']}")

        # Save prediction to file
        prediction_path = Path("test_prediction.json")
        with open(prediction_path, 'w') as f:
            json.dump(prediction, f, indent=2)
        logger.info(f"Prediction saved to {prediction_path}")

        return True
    except Exception as e:
        logger.error(f"Prediction test failed: {str(e)}")
        return False


def run_tests():
    """Run all tests."""
    test_results = {
        "health_check": test_health_check(),
        "prediction": test_prediction()
    }

    # Print summary
    logger.info("Test Results:")
    for test_name, result in test_results.items():
        logger.info(f"  {test_name}: {'✅ PASSED' if result else '❌ FAILED'}")

    # Return overall result
    return all(test_results.values())


if __name__ == "__main__":
    logger.info("Running API tests...")
    success = run_tests()
    sys.exit(0 if success else 1)
