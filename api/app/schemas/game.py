#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic schemas for game data (read-only).
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class GameResponse(BaseModel):
    """Schema for game responses."""
    game_id: str
    crash_point: Optional[float] = None
    hash_value: Optional[str] = None
    calculated_point: Optional[float] = None
    crashed_floor: Optional[int] = None
    end_time: Optional[datetime] = None
    prepare_time: Optional[datetime] = None
    begin_time: Optional[datetime] = None

    class Config:
        """Pydantic model config."""
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }
