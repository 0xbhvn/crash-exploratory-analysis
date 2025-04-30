#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Game model for crash game data.
"""

from sqlalchemy import Column, String, Float, Integer, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from ..database.db_config import Base


class Game(Base):
    """SQLAlchemy model for crash games."""

    __tablename__ = "crash_games"

    game_id = Column(String, primary_key=True, name='game_id')
    hash_value = Column(String, name='hash_value', nullable=True)
    crash_point = Column(Float, name='crash_point', nullable=True)
    calculated_point = Column(Float, name='calculated_point', nullable=True)
    crashed_floor = Column(Integer, name='crashed_floor', nullable=True)
    end_time = Column(DateTime, name='end_time', nullable=True)
    prepare_time = Column(DateTime, name='prepare_time', nullable=True)
    begin_time = Column(DateTime, name='begin_time', nullable=True)

    # Add indexes for commonly queried fields
    __table_args__ = (
        Index('ix_crash_games_crash_point', 'crash_point'),
        Index('ix_crash_games_begin_time', 'begin_time'),
        Index('ix_crash_games_end_time', 'end_time'),
    )

    def to_dict(self):
        """Convert model instance to dictionary."""
        return {
            'game_id': self.game_id,
            'hash_value': self.hash_value,
            'crash_point': float(self.crash_point) if self.crash_point is not None else None,
            'calculated_point': float(self.calculated_point) if self.calculated_point is not None else None,
            'crashed_floor': int(self.crashed_floor) if self.crashed_floor is not None else None,
            'end_time': self.end_time.isoformat() if self.end_time is not None else None,
            'prepare_time': self.prepare_time.isoformat() if self.prepare_time is not None else None,
            'begin_time': self.begin_time.isoformat() if self.begin_time is not None else None
        }
