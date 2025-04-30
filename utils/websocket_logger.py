#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlined WebSocket logger for Crash Game Streak Analysis.
Provides concise, informative logs for WebSocket data.
"""

import logging
import json
from typing import Dict, Any, Optional

from utils.logger_config import console, print_info, print_success, print_panel, print_warning

# Filter to prevent duplicate WebSocket message logging


class WebSocketFilter(logging.Filter):
    """Filter that prevents duplicate raw WebSocket logging."""

    def filter(self, record):
        """Filter out raw WebSocket messages if they'll be processed later."""
        if hasattr(record, 'is_raw_websocket') and record.is_raw_websocket:
            return False
        return True


# Create a specialized WebSocket logger
websocket_logger = logging.getLogger('websocket')
websocket_logger.addFilter(WebSocketFilter())

# Game status formatting


def format_game_status(game_id: str, crash_point: float) -> str:
    """Format game status with color-coded crash point."""

    crash_point_float = float(crash_point)
    if crash_point_float >= 10.0:
        return f"Game #{game_id} [bold bright_green]× {crash_point}[/bold bright_green]"
    elif crash_point_float >= 5.0:
        return f"Game #{game_id} [bright_yellow]× {crash_point}[/bright_yellow]"
    else:
        return f"Game #{game_id} × {crash_point}"


def log_game(game_data: Dict[str, Any]) -> None:
    """Log game information in a concise, formatted way."""

    try:
        # Extract game data - either direct or from nested 'data' structure
        if 'data' in game_data and isinstance(game_data['data'], dict):
            # Handle case where game_data is the full message with nested data
            game_info = game_data['data']
        else:
            # Handle case where game_data is already the game info
            game_info = game_data

        game_id = game_info.get('gameId')
        crash_point = game_info.get('crashPoint')

        if not game_id or not crash_point:
            print_warning(f"Missing required game data: {game_info}")
            return

        # Log only the essential information with proper formatting
        status = format_game_status(game_id, crash_point)
        console.print(status)

        # If this is a 10x or higher game, highlight it
        if float(crash_point) >= 10.0:
            # Create a more visually appealing 10x alert
            alert_msg = f"10× detected! Processing streak for game #{game_id}"
            console.print(
                f"[bold bright_green]{alert_msg}[/bold bright_green]")
    except KeyError as e:
        print_warning(f"Invalid game data format: {e}")
    except Exception as e:
        print_warning(f"Error logging game: {str(e)}")


def log_streak_processing(streak_number: int, game_range: str) -> None:
    """Log streak processing information concisely."""

    # Use a more visually distinctive formatting
    console.print(
        f"[bold bright_green]Saved streak #{streak_number} ({game_range})[/bold bright_green]")


def log_websocket_connection(status: str, url: Optional[str] = None) -> None:
    """Log WebSocket connection status."""

    if status == "connected" and url:
        print_panel(
            f"Connected to WebSocket\n{url}",
            title="WebSocket",
            style="green"
        )
    elif status == "reconnecting":
        print_warning("WebSocket connection lost. Reconnecting...")
    elif status == "closed":
        print_info("WebSocket connection closed")
