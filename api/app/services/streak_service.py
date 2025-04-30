#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Service for handling streak data.
"""

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..models.streak import Streak
from ..schemas.streak import StreakCreate
import pandas as pd


def create_streak(db: Session, streak: StreakCreate) -> Streak:
    """
    Create a new streak.

    Args:
        db: Database session
        streak: Streak data

    Returns:
        Created streak model
    """
    db_streak = Streak(
        streak_number=streak.streak_number,
        start_game_id=streak.start_game_id,
        end_game_id=streak.end_game_id,
        streak_length=streak.streak_length,
        hit_multiplier=streak.hit_multiplier,
        mean_multiplier=streak.mean_multiplier,
        std_multiplier=streak.std_multiplier,
        max_multiplier=streak.max_multiplier,
        min_multiplier=streak.min_multiplier
    )
    db.add(db_streak)
    db.commit()
    db.refresh(db_streak)
    return db_streak


def get_streaks(db: Session, skip: int = 0, limit: int = 100) -> List[Streak]:
    """
    Get streaks from the database.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of streaks
    """
    return db.query(Streak).order_by(Streak.streak_number.desc()).offset(skip).limit(limit).all()


def get_streak_by_id(db: Session, streak_id: int) -> Streak:
    """
    Get a streak by ID.

    Args:
        db: Database session
        streak_id: Streak ID

    Returns:
        Streak model
    """
    return db.query(Streak).filter(Streak.id == streak_id).first()


def get_last_streak(db: Session) -> Streak:
    """
    Get the last streak.

    Args:
        db: Database session

    Returns:
        Last streak model
    """
    return db.query(Streak).order_by(Streak.streak_number.desc()).first()


def process_games_for_streaks(games_df: pd.DataFrame, multiplier_threshold: float = 10.0) -> pd.DataFrame:
    """
    Process game data to extract streaks.

    Args:
        games_df: DataFrame with game data
        multiplier_threshold: Threshold for streak hit

    Returns:
        DataFrame with streak data
    """
    # Import processing function from main project
    from data_processing import extract_streaks_and_multipliers

    # Extract streaks
    streaks_df = extract_streaks_and_multipliers(
        games_df, multiplier_threshold)

    return streaks_df
