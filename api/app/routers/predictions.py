#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API router for prediction endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from ..database.db_config import get_db
from ..services import prediction_service
from ..schemas.prediction import PredictionResponse
from ..schemas.streak import StreakRequest
from ..models.prediction import Prediction

router = APIRouter(
    prefix="/predictions",
    tags=["predictions"]
)


@router.post("/", response_model=Dict[str, Any])
async def predict_next_streak(
    request: StreakRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Predict the next streak based on recent streak data.

    Returns:
        Prediction result
    """
    try:
        # Make prediction
        prediction = prediction_service.predict_next_streak(
            streak_data=request.recent_streaks,
            lookback=request.lookback
        )

        # Save prediction to database in the background
        background_tasks.add_task(
            prediction_service.save_prediction, db, prediction)

        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error predicting next streak: {str(e)}"
        )


@router.get("/", response_model=List[PredictionResponse])
async def get_predictions(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get recent predictions.

    Returns:
        List of predictions
    """
    predictions = prediction_service.get_predictions(db, skip, limit)
    return predictions


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(
    prediction_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a prediction by ID.

    Returns:
        Prediction
    """
    prediction = prediction_service.get_prediction_by_id(db, prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction with ID {prediction_id} not found"
        )
    return prediction
