#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic schemas for prediction data.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class PredictionBase(BaseModel):
    """Base schema for prediction data."""
    next_streak_number: int
    starts_after_game_id: int
    predicted_cluster: int
    prediction_desc: str
    confidence: float

    # Optional fields
    correct: Optional[bool] = None
    actual_streak_length: Optional[int] = None
    actual_cluster: Optional[int] = None

    # Confidence analysis
    confidence_distribution: Optional[Dict[str, float]] = None
    prediction_entropy: Optional[float] = None

    # Temporal context
    time_to_verification: Optional[int] = None

    # Model metadata
    model_version: Optional[str] = None
    feature_set: Optional[str] = None
    lookback_window: Optional[int] = None


class PredictionCreate(PredictionBase):
    """Schema for creating predictions."""
    pass


class PredictionResponse(PredictionBase):
    """Schema for prediction responses."""
    id: int
    created_at: str

    class Config:
        """Pydantic model config."""
        from_attributes = True


class PredictionUpdate(BaseModel):
    """Schema for updating prediction correctness."""
    correct: bool
    actual_streak_length: Optional[int] = None
    actual_cluster: Optional[int] = None
    time_to_verification: Optional[int] = None

    class Config:
        """Pydantic model config."""
        from_attributes = True
