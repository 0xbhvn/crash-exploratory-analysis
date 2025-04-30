#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction model for storing streak predictions.
"""

from sqlalchemy import Column, Integer, Float, String, DateTime, JSON, Boolean, func
from ..database.db_config import Base


class Prediction(Base):
    """SQLAlchemy model for streak predictions."""

    __tablename__ = "streak_predictions"

    id = Column(Integer, primary_key=True, index=True)
    next_streak_number = Column(Integer, nullable=False)
    starts_after_game_id = Column(Integer, nullable=False)
    predicted_cluster = Column(Integer, nullable=False)
    prediction_desc = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    prediction_data = Column(JSON, nullable=False)
    correct = Column(Boolean, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "next_streak_number": self.next_streak_number,
            "starts_after_game_id": self.starts_after_game_id,
            "predicted_cluster": self.predicted_cluster,
            "prediction_desc": self.prediction_desc,
            "confidence": self.confidence,
            "prediction_data": self.prediction_data,
            "correct": self.correct,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
