#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic schemas for prediction data.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
from pydantic import BaseModel, Field


class PredictionBase(BaseModel):
    """Base schema for prediction data."""
    next_streak_number: int
    starts_after_game_id: int
    predicted_cluster: int
    prediction_desc: str
    confidence: float
    prediction_data: Dict[str, Any]
    correct: Optional[bool] = None


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

    class Config:
        """Pydantic model config."""
        from_attributes = True
