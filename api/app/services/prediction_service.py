#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Service for handling streak predictions using the temporal model.
"""

import os
import pandas as pd
from typing import Dict, List, Any
from sqlalchemy.orm import Session
from pathlib import Path
import json

from ..models.prediction import Prediction
from ..schemas.prediction import PredictionCreate

# Import the prediction function from the main project
from temporal.deploy import load_model_and_predict

# Model paths - check multiple locations
ROOT_DIR = Path(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../..')))
API_DIR = Path(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')))
MODEL_PATHS = [
    ROOT_DIR / "output" / "temporal_model.pkl",  # project root/output/
    API_DIR / "output" / "temporal_model.pkl",   # api/output/
    Path("/models/temporal_model.pkl"),          # Docker volume
    # Relative to current working dir
    Path("output/temporal_model.pkl"),
]

# Find first valid model path
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    raise FileNotFoundError(
        f"Could not find temporal_model.pkl in any of the expected locations: {[str(p) for p in MODEL_PATHS]}")


def predict_next_streak(streak_data: List[Dict[str, Any]], lookback: int = 50) -> Dict[str, Any]:
    """
    Make a prediction for the next streak.

    Args:
        streak_data: List of streak dictionaries
        lookback: Number of recent streaks to use

    Returns:
        Prediction result
    """
    # Create DataFrame from streak data
    df = pd.DataFrame(streak_data)

    # Use only the most recent streaks based on lookback parameter
    recent_streaks = df.tail(lookback)

    # Make prediction using the temporal model
    try:
        prediction = load_model_and_predict(MODEL_PATH, recent_streaks)
        return prediction
    except Exception as e:
        raise RuntimeError(f"Error predicting next streak: {str(e)}")


def save_prediction(db: Session, prediction: Dict[str, Any]) -> Prediction:
    """
    Save a prediction to the database.

    Args:
        db: Database session
        prediction: Prediction dictionary

    Returns:
        Saved prediction model
    """
    # Create prediction model
    prediction_data = PredictionCreate(
        next_streak_number=prediction["next_streak_number"],
        starts_after_game_id=prediction["starts_after_game_id"],
        predicted_cluster=prediction["predicted_cluster"],
        prediction_desc=prediction["prediction_desc"],
        confidence=prediction["confidence"],
        prediction_data=prediction
    )

    # Create DB model
    db_prediction = Prediction(
        next_streak_number=prediction_data.next_streak_number,
        starts_after_game_id=prediction_data.starts_after_game_id,
        predicted_cluster=prediction_data.predicted_cluster,
        prediction_desc=prediction_data.prediction_desc,
        confidence=prediction_data.confidence,
        prediction_data=prediction_data.prediction_data
    )

    # Save to database
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)

    return db_prediction


def get_predictions(db: Session, skip: int = 0, limit: int = 100) -> List[Prediction]:
    """
    Get predictions from the database.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of predictions
    """
    return db.query(Prediction).order_by(Prediction.id.desc()).offset(skip).limit(limit).all()


def get_prediction_by_id(db: Session, prediction_id: int) -> Prediction:
    """
    Get a prediction by ID.

    Args:
        db: Database session
        prediction_id: Prediction ID

    Returns:
        Prediction model
    """
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()
