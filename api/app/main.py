#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main FastAPI application for Crash Game Streak Prediction API.
"""

import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from .database.db_config import engine, Base
from .routers import games_router, streaks_router, predictions_router, websocket_router
from .routers.websocket import connect_to_external_websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Startup and shutdown tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup task - connect to external websocket
    websocket_task = asyncio.create_task(connect_to_external_websocket())

    yield

    # Shutdown tasks
    websocket_task.cancel()
    try:
        await websocket_task
    except asyncio.CancelledError:
        logger.info("External websocket connection cancelled")

# Create FastAPI app
app = FastAPI(
    title="Crash Game Streak Prediction API",
    description="API for predicting 10× streaks in crash games",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Include routers
app.include_router(games_router)
app.include_router(streaks_router)
app.include_router(predictions_router)
app.include_router(websocket_router)


@app.get("/", tags=["health"])
async def root():
    """
    API health check endpoint.

    Returns:
        Status message
    """
    return {"status": "healthy", "message": "Crash Game Streak Prediction API is running"}
