#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WebSocket API for real-time game updates.
"""

import asyncio
import json
import logging
import pandas as pd
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from typing import Dict, List, Set, Any
import websockets

from ..database.db_config import get_db
from ..services import game_service, streak_service, prediction_service
from ..schemas.game import GameCreate
from ..schemas.streak import StreakCreate

# Set up logger
logger = logging.getLogger(__name__)

# Router
router = APIRouter(tags=["websocket"])

# WebSocket connections manager


class ConnectionManager:
    """
    Manager for WebSocket connections.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.games_buffer: List[Dict[str, Any]] = []
        self.multiplier_threshold = 10.0

    async def connect(self, websocket: WebSocket):
        """
        Connect a WebSocket client.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket client.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """
        Broadcast a message to all connected clients.
        """
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {str(e)}")

    def add_game_to_buffer(self, game: Dict[str, Any]):
        """
        Add a game to the buffer.
        """
        self.games_buffer.append(game)

    def clear_buffer(self):
        """
        Clear the games buffer.
        """
        self.games_buffer = []

    def get_buffer_as_dataframe(self):
        """
        Get the games buffer as a DataFrame.
        """
        if not self.games_buffer:
            return pd.DataFrame()

        df = pd.DataFrame(self.games_buffer)
        if "game_id" in df.columns and "crash_point" in df.columns:
            df = df.rename(
                columns={"game_id": "Game ID", "crash_point": "Bust"})
        return df


# Create connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    """
    WebSocket endpoint for real-time updates.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Parse received game data
                game_data = json.loads(data)

                # Save game to database
                game = GameCreate(
                    game_id=game_data["game_id"],
                    crash_point=game_data["crash_point"]
                )
                game_service.create_game(db, game)

                # Add game to buffer
                manager.add_game_to_buffer(game_data)

                # Process buffer for streaks when we hit a 10x or higher
                if game_data["crash_point"] >= manager.multiplier_threshold:
                    games_df = manager.get_buffer_as_dataframe()
                    if not games_df.empty:
                        # Process streaks
                        streaks_df = streak_service.process_games_for_streaks(
                            games_df, manager.multiplier_threshold
                        )

                        # Save the last streak
                        if not streaks_df.empty:
                            last_streak = streaks_df.iloc[-1].to_dict()
                            streak = StreakCreate(**last_streak)
                            db_streak = streak_service.create_streak(
                                db, streak)

                            # Predict the next streak
                            prediction = prediction_service.predict_next_streak(
                                streak_data=streaks_df.to_dict('records'),
                                lookback=50
                            )

                            # Save prediction
                            prediction_service.save_prediction(db, prediction)

                            # Broadcast prediction to all clients
                            await manager.broadcast({
                                "type": "prediction",
                                "prediction": prediction
                            })

                # Broadcast game data to all clients
                await manager.broadcast({
                    "type": "game",
                    "game": game_data
                })

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {data}")
            except Exception as e:
                logger.error(f"Error processing game data: {str(e)}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Connect to external websocket


async def connect_to_external_websocket():
    """
    Connect to external WebSocket for game data.
    """
    uri = "wss://crashed-proxy-production.up.railway.app/ws"
    async for websocket in websockets.connect(uri):
        try:
            logger.info(f"Connected to external websocket: {uri}")
            async for message in websocket:
                try:
                    # Parse received game data
                    game_data = json.loads(message)

                    # Add game to buffer
                    manager.add_game_to_buffer(game_data)

                    # Broadcast game data to all clients
                    await manager.broadcast({
                        "type": "game",
                        "game": game_data
                    })

                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from external: {message}")
                except Exception as e:
                    logger.error(f"Error processing external data: {str(e)}")
        except websockets.ConnectionClosed:
            logger.info(
                "Connection to external websocket closed. Reconnecting...")
            continue
