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
    correct = Column(Boolean, nullable=True)

    # Ground truth fields
    actual_streak_length = Column(Integer, nullable=True)
    actual_cluster = Column(Integer, nullable=True)

    # Confidence analysis
    confidence_distribution = Column(
        JSON, nullable=True)  # All class probabilities
    prediction_entropy = Column(Float, nullable=True)  # Uncertainty measure

    # Temporal context
    # Seconds between prediction and streak completion
    time_to_verification = Column(Integer, nullable=True)

    # Model metadata
    model_version = Column(String, nullable=True)
    feature_set = Column(String, nullable=True)
    lookback_window = Column(Integer, nullable=True)

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
            "correct": self.correct,
            "actual_streak_length": self.actual_streak_length,
            "actual_cluster": self.actual_cluster,
            "confidence_distribution": self.confidence_distribution,
            "prediction_entropy": self.prediction_entropy,
            "time_to_verification": self.time_to_verification,
            "model_version": self.model_version,
            "feature_set": self.feature_set,
            "lookback_window": self.lookback_window,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
