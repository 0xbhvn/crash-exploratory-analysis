#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Service for handling game data (read-only).
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..models.game import Game


def get_games(db: Session, skip: int = 0, limit: int = 100) -> List[Game]:
    """
    Get games from the database.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return

    Returns:
        List of games
    """
    return db.query(Game).order_by(desc(Game.game_id)).offset(skip).limit(limit).all()


def get_game_by_id(db: Session, game_id: str) -> Optional[Game]:
    """
    Get a game by ID.

    Args:
        db: Database session
        game_id: Game ID

    Returns:
        Game model or None
    """
    return db.query(Game).filter(Game.game_id == game_id).first()


def get_last_game_id(db: Session) -> Optional[str]:
    """
    Get the last game ID.

    Args:
        db: Database session

    Returns:
        Last game ID or None
    """
    last_game = db.query(Game).order_by(desc(Game.game_id)).first()
    return last_game.game_id if last_game else None
