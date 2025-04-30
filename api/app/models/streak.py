#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streak model for storing streak data.
"""

from sqlalchemy import Column, Integer, Float, DateTime, func, Index
from sqlalchemy.sql import expression
from ..database.db_config import Base


class Streak(Base):
    """SQLAlchemy model for game streaks."""

    __tablename__ = "streaks"

    id = Column(Integer, primary_key=True, index=True)
    streak_number = Column(Integer, unique=True, nullable=False, index=True)
    start_game_id = Column(Integer, nullable=False)
    end_game_id = Column(Integer, nullable=False)
    streak_length = Column(Integer, nullable=False)
    hit_multiplier = Column(Float, nullable=False)
    mean_multiplier = Column(Float, nullable=False)
    std_multiplier = Column(Float, nullable=True)
    max_multiplier = Column(Float, nullable=False)
    min_multiplier = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Add composite index for start_game_id and end_game_id
    __table_args__ = (
        Index('ix_streaks_game_range', 'start_game_id', 'end_game_id'),
    )

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "streak_number": self.streak_number,
            "start_game_id": self.start_game_id,
            "end_game_id": self.end_game_id,
            "streak_length": self.streak_length,
            "hit_multiplier": self.hit_multiplier,
            "mean_multiplier": self.mean_multiplier,
            "std_multiplier": self.std_multiplier,
            "max_multiplier": self.max_multiplier,
            "min_multiplier": self.min_multiplier,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
