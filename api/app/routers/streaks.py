#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API router for streak endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..database.db_config import get_db
from ..services import streak_service
from ..schemas.streak import StreakResponse, StreakCreate

router = APIRouter(
    prefix="/streaks",
    tags=["streaks"]
)


@router.post("/", response_model=StreakResponse)
async def create_streak(
    streak: StreakCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new streak.

    Returns:
        Created streak
    """
    return streak_service.create_streak(db, streak)


@router.get("/", response_model=List[StreakResponse])
async def get_streaks(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Get recent streaks.

    Returns:
        List of streaks
    """
    streaks = streak_service.get_streaks(db, skip, limit)
    return streaks


@router.get("/last", response_model=StreakResponse)
async def get_last_streak(
    db: Session = Depends(get_db)
):
    """
    Get the last streak.

    Returns:
        Last streak
    """
    streak = streak_service.get_last_streak(db)
    if not streak:
        raise HTTPException(
            status_code=404,
            detail="No streaks found"
        )
    return streak


@router.get("/{streak_id}", response_model=StreakResponse)
async def get_streak(
    streak_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a streak by ID.

    Returns:
        Streak
    """
    streak = streak_service.get_streak_by_id(db, streak_id)
    if not streak:
        raise HTTPException(
            status_code=404,
            detail=f"Streak with ID {streak_id} not found"
        )
    return streak
