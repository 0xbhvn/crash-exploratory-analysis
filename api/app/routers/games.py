#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API router for game endpoints (read-only).
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from ..database.db_config import get_db
from ..services import game_service
from ..schemas.game import GameResponse

router = APIRouter(
    prefix="/games",
    tags=["games"]
)


@router.get("/", response_model=List[GameResponse])
async def get_games(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get recent games.

    Returns:
        List of games
    """
    games = game_service.get_games(db, skip, limit)
    return games


@router.get("/last_id", response_model=Optional[str])
async def get_last_game_id(
    db: Session = Depends(get_db)
):
    """
    Get the last game ID.

    Returns:
        Last game ID or None
    """
    return game_service.get_last_game_id(db)


@router.get("/{game_id}", response_model=GameResponse)
async def get_game(
    game_id: str,
    db: Session = Depends(get_db)
):
    """
    Get a game by ID.

    Returns:
        Game
    """
    game = game_service.get_game_by_id(db, game_id)
    if not game:
        raise HTTPException(
            status_code=404,
            detail=f"Game with ID {game_id} not found"
        )
    return game
