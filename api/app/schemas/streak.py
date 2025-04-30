#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic schemas for streak data.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class StreakBase(BaseModel):
    """Base schema for streak data."""
    streak_number: int
    start_game_id: int
    end_game_id: int
    streak_length: int
    hit_multiplier: float
    mean_multiplier: float
    std_multiplier: Optional[float] = None
    max_multiplier: float
    min_multiplier: float


class StreakCreate(StreakBase):
    """Schema for creating streaks."""
    pass


class StreakResponse(StreakBase):
    """Schema for streak responses."""
    id: int
    created_at: str

    class Config:
        """Pydantic model config."""
        from_attributes = True


class StreakRequest(BaseModel):
    """Schema for streak prediction request."""
    recent_streaks: List[Dict[str, Any]]
    lookback: Optional[int] = Field(
        default=50, description="Number of recent streaks to use")
