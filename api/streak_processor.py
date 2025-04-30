#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script for continuous streak processing and prediction.

This script runs perpetually to:
1. Fetch new crash game data
2. Process data into streaks
3. Make predictions for upcoming streaks
4. Update previous predictions with correctness results

The script has two main modes:
- Catchup mode: Process all historical data until reaching the latest
- Continuous mode: Continuously monitor for new data and process incrementally
"""

from utils.logger_config import (
    console, print_info, print_success, print_warning, print_error, print_panel
)
from utils.websocket_logger import (
    log_game, log_streak_processing, log_websocket_connection,
    websocket_logger
)
import os
import time
import logging
import pandas as pd
import argparse
from sqlalchemy import create_engine, text, desc, func
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
import json
import requests
import sys
from datetime import datetime, timedelta
import csv
import asyncio
import websockets
import ssl

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Add paths to sys.path
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

# Import websocket logger and rich utilities

# Configure logging for file-based logs
# Ensure logs directory exists
os.makedirs(os.path.join(current_dir, "logs"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(
            current_dir, "logs/streak_processor.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Apply WebSocket filter to root logger to prevent duplicate raw WebSocket logs
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        # Add filter from websocket_logger to stdout handler only
        handler.addFilter(websocket_logger.filters[0])

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description="Continuous streak processor")
parser.add_argument("--test", action="store_true",
                    help="Run in test mode (fetch one batch then exit)")
parser.add_argument("--catchup", action="store_true",
                    help="Catchup all historical data first")
parser.add_argument("--interval", type=int, default=None,
                    help="Override fetch interval in seconds")
parser.add_argument("--websocket", action="store_true",
                    help="Use WebSocket for real-time updates instead of polling")
args = parser.parse_args()

# Constants
MULTIPLIER_THRESHOLD = float(os.getenv("MULTIPLIER_THRESHOLD", "10.0"))
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", "60")
                     ) if args.interval is None else args.interval
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(
    project_root, "output/temporal_model.pkl"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
WEBSOCKET_URI = os.getenv(
    "WEBSOCKET_URI", "wss://crashed-proxy-production.up.railway.app/ws")

# Set log level
logging.getLogger().setLevel(getattr(logging, LOG_LEVEL))

# Display compact configuration panel instead of verbose logging
config_info = [
    f"Threshold: {MULTIPLIER_THRESHOLD}×",
    f"Connection: {'WebSocket' if args.websocket else 'Polling'}",
    f"Mode: {'Test' if args.test else 'Production'}"
]
print_panel("\n".join(config_info), title="Streak Processor", style="blue")

# Try to import dependent modules
try:
    # First try temporal from parent project root
    sys.path.insert(0, project_root)
    from temporal.deploy import load_model_and_predict
    from data_processing import extract_streaks_and_multipliers

    # Then try app modules
    from app.database.db_config import Base
    from app.models.streak import Streak
    from app.models.prediction import Prediction
    from app.models.game import Game

    logger.debug("Successfully imported all required modules")
except ImportError as e:
    print_error(f"Failed to import modules: {str(e)}")
    logger.error(f"Failed to import modules: {str(e)}")
    logger.error(f"sys.path: {sys.path}")
    raise

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

logger.debug(
    f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'DB'}")

try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    # Test the connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        if result.scalar() == 1:
            logger.debug("Database connection test successful")
        else:
            print_error("Database connection test failed")
except Exception as e:
    print_error(f"Error connecting to database: {str(e)}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Game buffer for WebSocket mode - removed as we now use database queries
# games_buffer = []


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        raise


def fetch_latest_games(last_game_id: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch the latest games from the database.

    Args:
        last_game_id: Last processed game ID

    Returns:
        DataFrame with game data
    """
    logger.debug(f"Fetching latest games after ID {last_game_id}")

    db = get_db()
    try:
        query = db.query(Game)

        if last_game_id:
            query = query.filter(Game.game_id > last_game_id)

        query = query.order_by(Game.game_id)

        games = query.all()

        if not games:
            logger.debug("No new games found")
            return pd.DataFrame()

        # Convert to DataFrame
        games_data = []
        for game in games:
            game_dict = {
                "game_id": game.game_id,
                "crash_point": game.crash_point,
                "end_time": game.end_time
            }
            games_data.append(game_dict)

        df = pd.DataFrame(games_data)
        logger.debug(f"Fetched {len(df)} new games")
        return df
    finally:
        db.close()


def process_streaks(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process game data to identify streaks.

    Args:
        games_df: DataFrame with game data

    Returns:
        DataFrame with streak data
    """
    if games_df.empty:
        return pd.DataFrame()

    logger.debug(f"Processing {len(games_df)} games for streaks")

    # Rename columns to match expected format
    games_df = games_df.rename(columns={
        'game_id': 'Game ID',
        'crash_point': 'Bust'
    })

    # Extract streaks
    streaks_df = extract_streaks_and_multipliers(
        games_df, MULTIPLIER_THRESHOLD)

    if not streaks_df.empty:
        # Use rich formatting for streak extraction message
        console.print(
            f"[bright_cyan]Extracted {len(streaks_df)} streaks[/bright_cyan]")
    else:
        console.print("[bright_cyan]No streaks extracted[/bright_cyan]")

    return streaks_df


def save_streaks(streaks_df: pd.DataFrame) -> List[int]:
    """
    Save streaks to the database with optimized queries.

    Args:
        streaks_df: DataFrame with streak data

    Returns:
        List of saved streak IDs
    """
    if streaks_df.empty:
        logger.debug("No streaks to save, returning empty list")
        return []

    logger.debug(f"Saving {len(streaks_df)} streaks to database")

    db = get_db()
    saved_ids = []
    saved_count = 0
    already_existing = 0
    failed_streaks = 0

    try:
        # Use a single transaction for all operations
        with db.begin():
            # Get current max streak number for proper numbering
            max_streak_number = db.query(
                func.max(Streak.streak_number)).scalar() or 0
            logger.debug(f"Current max streak number: {max_streak_number}")

            for idx, row in streaks_df.iterrows():
                try:
                    # Check for duplicate using EXISTS query (much more efficient)
                    from sqlalchemy.sql import exists
                    streak_exists = db.query(exists().where(
                        (Streak.streak_number == row['streak_number']) |
                        ((Streak.start_game_id == int(row['start_game_id'])) &
                         (Streak.end_game_id == int(row['end_game_id'])))
                    )).scalar()

                    if streak_exists:
                        already_existing += 1
                        continue

                    # Create new streak
                    streak = Streak(
                        streak_number=int(row['streak_number']),
                        start_game_id=int(row['start_game_id']),
                        end_game_id=int(row['end_game_id']),
                        streak_length=int(row['streak_length']),
                        hit_multiplier=float(row['hit_multiplier']),
                        mean_multiplier=float(row['mean_multiplier']),
                        std_multiplier=float(row['std_multiplier']) if 'std_multiplier' in row and pd.notna(
                            row['std_multiplier']) else None,
                        max_multiplier=float(row['max_multiplier']),
                        min_multiplier=float(row['min_multiplier'])
                    )

                    # Add the streak to the session
                    db.add(streak)
                    db.flush()  # Get ID without committing

                    saved_ids.append(streak.id)
                    saved_count += 1

                    # Log streak info
                    game_range = f"{row['start_game_id']}-{row['end_game_id']}"
                    log_streak_processing(row['streak_number'], game_range)

                except Exception as e:
                    failed_streaks += 1
                    logger.error(
                        f"Error saving streak #{row.get('streak_number', 'unknown')}: {str(e)}")
                    # Individual streak error doesn't roll back entire transaction

            # Automatic commit at end of with block

        # Print summary if any activity
        if saved_count > 0 or failed_streaks > 0:
            stats = {
                "New streaks": saved_count,
                "Already existed": already_existing,
                "Failed": failed_streaks
            }
            from utils.logger_config import create_stats_table
            create_stats_table("Streak Processing", stats)

        return saved_ids

    except Exception as e:
        print_error(f"Fatal error in save_streaks: {str(e)}")
        logger.error(f"Fatal error in save_streaks: {str(e)}")
        raise
    finally:
        db.close()


def update_prediction_correctness() -> int:
    """
    Update the correctness of previous predictions based on actual results.

    Returns:
        Number of updated predictions
    """
    db = get_db()
    updated_count = 0

    try:
        # Find predictions that don't have correctness values yet
        pending_predictions = db.query(Prediction).filter(
            Prediction.correct.is_(None)
        ).all()

        if not pending_predictions:
            logger.debug("No pending predictions to update")
            return 0

        logger.debug(
            f"Found {len(pending_predictions)} pending predictions to evaluate")

        # Process each pending prediction
        for prediction in pending_predictions:
            try:
                # Get the actual streak for this prediction
                actual_streak = db.query(Streak).filter(
                    Streak.streak_number == prediction.next_streak_number
                ).first()

                if not actual_streak:
                    # Streak hasn't occurred yet
                    continue

                # Store the actual streak length
                actual_length = actual_streak.streak_length
                prediction.actual_streak_length = actual_length

                # Determine the actual cluster based on streak length
                actual_cluster = None

                if actual_length <= 3:
                    actual_cluster = 0  # short
                elif actual_length <= 7:
                    actual_cluster = 1  # medium_short
                elif actual_length <= 14:
                    actual_cluster = 2  # medium_long
                else:
                    actual_cluster = 3  # long

                # Store the actual cluster
                prediction.actual_cluster = actual_cluster

                # Update prediction correctness
                prediction.correct = (
                    prediction.predicted_cluster == actual_cluster)

                # Calculate time to verification if possible
                if prediction.created_at and actual_streak.created_at:
                    time_diff = actual_streak.created_at - prediction.created_at
                    prediction.time_to_verification = int(
                        time_diff.total_seconds())

                updated_count += 1

                logger.debug(
                    f"Updated prediction #{prediction.id} for streak #{prediction.next_streak_number}: "
                    f"predicted={prediction.predicted_cluster} ({prediction.prediction_desc}), "
                    f"actual={actual_cluster} (length={actual_length}), "
                    f"correct={prediction.correct}"
                )

            except Exception as e:
                logger.error(
                    f"Error updating prediction #{prediction.id}: {str(e)}")
                # Continue with other predictions

        # Commit all updates
        if updated_count > 0:
            db.commit()

        return updated_count

    except Exception as e:
        db.rollback()
        print_error(f"Error in update_prediction_correctness: {str(e)}")
        logger.error(f"Error in update_prediction_correctness: {str(e)}")
        return 0
    finally:
        db.close()


def save_prediction(streak_number: int, prediction: Dict[str, Any], db=None) -> bool:
    """
    Save a streak prediction to the database.

    Args:
        streak_number: The streak number to predict after
        prediction: Prediction dictionary from the model
        db: Database session (optional, will create one if not provided)

    Returns:
        Boolean indicating success
    """
    close_db = False
    if db is None:
        db = get_db()
        close_db = True

    try:
        # Create prediction model with enhanced fields
        pred = Prediction(
            next_streak_number=prediction["next_streak_number"],
            starts_after_game_id=prediction["starts_after_game_id"],
            predicted_cluster=prediction["predicted_cluster"],
            prediction_desc=prediction["prediction_desc"],
            confidence=prediction["confidence"],
            correct=None,  # Will be updated later when the streak occurs

            # Include enhanced analytics fields
            confidence_distribution=prediction.get("confidence_distribution"),
            prediction_entropy=prediction.get("prediction_entropy"),
            model_version=prediction.get("model_version"),
            feature_set=prediction.get("feature_set"),
            lookback_window=prediction.get("lookback_window")
        )

        # Save to database
        db.add(pred)
        db.commit()

        print_success(
            f"Saved prediction for streak #{prediction['next_streak_number']} (after game #{prediction['starts_after_game_id']}): "
            f"{prediction['prediction_desc']} with {prediction['confidence']:.4f} confidence"
        )

        # Print the prediction with rich formatting
        from utils.logger_config import create_table, add_table_row, display_table

        pred_table = create_table(
            f"Prediction for Next Streak #{prediction['next_streak_number']}",
            ["Attribute", "Value"]
        )

        add_table_row(pred_table, ["Predicted Length",
                      prediction['prediction_desc']])
        add_table_row(pred_table, ["Confidence",
                      f"{prediction['confidence']:.4f}"])
        add_table_row(pred_table, ["After Game ID", str(
            prediction['starts_after_game_id'])])
        if prediction.get("prediction_entropy") is not None:
            add_table_row(pred_table, [
                          "Uncertainty", f"{prediction['prediction_entropy']:.4f} (lower is better)"])
        if prediction.get("model_version") is not None:
            add_table_row(pred_table, ["Model", prediction['model_version']])

        display_table(pred_table)

        return True

    except Exception as e:
        db.rollback()
        print_error(f"Error saving prediction: {str(e)}")
        logger.error(f"Error saving prediction: {str(e)}")
        return False
    finally:
        if close_db:
            db.close()


def make_prediction(streaks_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Make a prediction for the next streak using the temporal model.

    Args:
        streaks_df: DataFrame with streak data

    Returns:
        Dictionary with prediction details
    """
    if streaks_df.empty:
        logger.warning("Cannot make prediction: No streak data provided")
        return {}

    try:
        # Make sure the model path exists
        if not os.path.exists(MODEL_PATH):
            print_error(f"Model file not found at {MODEL_PATH}")
            available_dirs = [d for d in os.listdir(
                project_root) if os.path.isdir(os.path.join(project_root, d))]
            logger.error(
                f"Available directories in project root: {available_dirs}")
            return {}

        # Use debug level for verbose information
        logger.debug(f"Making prediction using model at {MODEL_PATH}")
        logger.debug(f"Using {len(streaks_df)} streaks for prediction")

        # Use temporal model to predict the next streak
        prediction = load_model_and_predict(MODEL_PATH, streaks_df)

        if not prediction:
            print_error("Prediction failed: Model returned empty result")
            return {}

        logger.debug(f"Raw prediction: {prediction}")

        # Enhance prediction with additional analytics data
        prediction["model_version"] = os.path.basename(MODEL_PATH)
        # Set your actual feature set name
        prediction["feature_set"] = "temporal"
        prediction["lookback_window"] = 5  # Set your actual lookback window

        # Store full confidence distribution
        if "prob_class_0" in prediction and "prob_class_1" in prediction and "prob_class_2" in prediction and "prob_class_3" in prediction:
            prediction["confidence_distribution"] = {
                "class_0": float(prediction["prob_class_0"]),
                "class_1": float(prediction["prob_class_1"]),
                "class_2": float(prediction["prob_class_2"]),
                "class_3": float(prediction["prob_class_3"])
            }

            # Calculate prediction entropy (higher means more uncertain)
            import math
            probs = [
                prediction["prob_class_0"],
                prediction["prob_class_1"],
                prediction["prob_class_2"],
                prediction["prob_class_3"]
            ]
            entropy = 0
            for p in probs:
                if p > 0:
                    entropy -= p * math.log2(p)
            prediction["prediction_entropy"] = float(entropy)

        # Return the prediction
        return prediction

    except Exception as e:
        print_error(f"Error making prediction: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        return {}


def get_streak_info() -> Tuple[Optional[int], Optional[str]]:
    """
    Get information about streaks in the database.

    Returns:
        Tuple containing:
        - Last streak number
        - Last processed game ID
    """
    db = get_db()
    try:
        # Get the most recent streak
        latest_streak = db.query(Streak).order_by(
            Streak.streak_number.desc()).first()

        if latest_streak:
            return latest_streak.streak_number, str(latest_streak.end_game_id)
        return None, None
    finally:
        db.close()


def get_first_unprocessed_game_id() -> Optional[str]:
    """Get the first game ID that hasn't been processed into streaks."""
    db = get_db()
    try:
        # Get minimum game ID
        first_game = db.query(func.min(Game.game_id)).scalar()
        if not first_game:
            return None

        # Get the last processed game
        latest_streak = db.query(Streak).order_by(
            Streak.end_game_id.desc()).first()

        if not latest_streak:
            # No streaks yet, start from the beginning
            return first_game

        # Start from the game after the last processed game
        return str(int(latest_streak.end_game_id) + 1)
    finally:
        db.close()


def get_min_max_game_ids() -> Tuple[Optional[str], Optional[str]]:
    """Get the minimum and maximum game IDs in the database."""
    db = get_db()
    try:
        min_game_id = db.query(func.min(Game.game_id)).scalar()
        max_game_id = db.query(func.max(Game.game_id)).scalar()
        return min_game_id, max_game_id
    finally:
        db.close()


def count_games() -> int:
    """Count the total number of games in the database."""
    db = get_db()
    try:
        return db.query(Game).count()
    finally:
        db.close()


def count_streaks() -> int:
    """Count the total number of streaks in the database."""
    db = get_db()
    try:
        return db.query(Streak).count()
    finally:
        db.close()


def get_all_streaks() -> pd.DataFrame:
    """Get all streaks from the database as a DataFrame."""
    db = get_db()
    try:
        streaks = db.query(Streak).order_by(Streak.streak_number).all()
        streaks_data = [s.to_dict() for s in streaks]
        return pd.DataFrame(streaks_data)
    finally:
        db.close()


async def process_websocket_game(message_data: Dict[str, Any]) -> None:
    """
    Process a single message received from WebSocket.

    Args:
        message_data: Message data dictionary from WebSocket
    """
    try:
        # Check if this is a connection message - nothing to process
        if message_data.get('type') == 'connection_established':
            # Skip logging - the WebSocket panel already shows we're connected
            return

        # Extract game data from the appropriate structure
        # Format: {"type": "new_game", "data": {"gameId": "123", "crashPoint": 1.5, ...}}
        if message_data.get('type') == 'new_game' and 'data' in message_data:
            game_data = message_data['data']

            # Create a standardized game info dictionary
            game_info = {
                'game_id': str(game_data.get('gameId')),
                'crash_point': float(game_data.get('crashPoint')),
                'end_time': game_data.get('endTime')
            }

            # Use the websocket logger to log game info
            log_game(game_data)

            # Add game to database
            db = get_db()
            try:
                # Check if game already exists
                existing_game = db.query(Game).filter(
                    Game.game_id == game_info["game_id"]).first()
                if not existing_game:
                    # Create new game
                    game = Game(
                        game_id=game_info["game_id"],
                        crash_point=game_info["crash_point"],
                        end_time=datetime.now() if not game_info.get(
                            'end_time') else game_info['end_time']
                    )
                    db.add(game)
                    db.commit()
                else:
                    logger.debug(
                        f"Game #{game_info['game_id']} already exists, skipping")
            except Exception as e:
                db.rollback()
                print_error(f"Error saving game to database: {str(e)}")
            finally:
                db.close()

            # Process streaks if this game is a 10x or higher
            if float(game_info["crash_point"]) >= MULTIPLIER_THRESHOLD:
                await process_streaks_from_db()
        else:
            logger.warning(f"Unrecognized message format: {message_data}")
    except Exception as e:
        print_error(f"Error processing WebSocket message: {str(e)}")
        logger.error(f"Data that caused error: {message_data}")


async def process_streaks_from_db() -> None:
    """
    Process streaks from the database efficiently.
    This queries all games since the last processed streak's end_game_id.
    """
    logger.debug("Processing streaks from database")

    try:
        db = get_db()
        try:
            # In a single query, get:
            # 1. The last processed streak's end_game_id
            # 2. The current max streak number
            last_streak = db.query(Streak).order_by(
                Streak.end_game_id.desc()).first()
            max_streak_number = db.query(
                func.max(Streak.streak_number)).scalar() or 0

            if last_streak:
                last_processed_game_id = last_streak.end_game_id
                logger.debug(
                    f"Last processed game ID: {last_processed_game_id}")
            else:
                # If no streaks yet, get the first game ID
                first_game = db.query(func.min(Game.game_id)).scalar()
                last_processed_game_id = int(
                    first_game) - 1 if first_game else 0
                logger.debug(
                    f"No streaks in database, starting from: {last_processed_game_id}")

            # Get all games since the last processed streak - use existing Game schema
            # Note: Game.game_id is a string in the production DB
            query = db.query(Game).filter(
                Game.game_id > str(last_processed_game_id)
            ).order_by(Game.game_id)

            games = query.all()

            if not games:
                logger.debug("No new games to process")
                return

            # Convert to DataFrame efficiently using existing Game schema
            games_data = [
                {
                    "game_id": game.game_id,  # String in production DB
                    "crash_point": float(game.crash_point) if game.crash_point is not None else 0.0,
                    "end_time": game.end_time
                }
                for game in games
            ]
            games_df = pd.DataFrame(games_data)

            logger.debug(
                f"Processing {len(games_df)} games (IDs {games_df['game_id'].min()}-{games_df['game_id'].max()})")
        finally:
            db.close()

        # Process the games into streaks
        streaks_df = process_streaks(games_df)

        if streaks_df.empty:
            logger.debug("No new streaks found")
            return

        # Adjust streak numbers to continue from the max
        streaks_df['streak_number'] = range(
            max_streak_number + 1, max_streak_number + 1 + len(streaks_df))

        # Save the streaks
        saved_ids = save_streaks(streaks_df)

        # If we saved any new streaks, make a prediction for the next streak
        if saved_ids:
            # Get all streaks for use in making prediction
            all_streaks_df = get_all_streaks()

            # Make prediction for the next streak and save in a single workflow
            prediction = make_prediction(all_streaks_df)

            # Save prediction if valid
            if prediction and 'next_streak_number' in prediction:
                db = get_db()
                try:
                    save_prediction(max_streak_number +
                                    len(streaks_df), prediction, db)

                    # Update correctness of previous predictions
                    updated_count = update_prediction_correctness()
                    if updated_count > 0:
                        print_info(
                            f"Updated {updated_count} previous prediction(s) with actual results")
                finally:
                    db.close()

    except Exception as e:
        print_error(f"Error processing streaks: {str(e)}")
        logger.error("Stack trace:", exc_info=True)


async def websocket_continuous_processing() -> None:
    """
    Run continuous processing using WebSocket for real-time updates.
    Connects to the external WebSocket and processes games as they arrive.
    """
    # First do a quick catchup for any missed games
    _, last_game_id = get_streak_info()
    await quick_catchup(last_game_id)

    # SSL context for WebSocket connection
    ssl_context = ssl.create_default_context()

    # For development only: if on macOS and getting certificate errors, we can disable verification
    # Note: This is not secure for production environments
    if sys.platform == 'darwin':  # macOS
        print_warning("Using relaxed SSL verification for development (macOS)")
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    # Then connect to WebSocket for real-time updates
    reconnect_delay = 5  # seconds
    while True:
        try:
            async with websockets.connect(WEBSOCKET_URI, ssl=ssl_context) as websocket:
                # Log successful connection
                log_websocket_connection("connected", WEBSOCKET_URI)

                async for message in websocket:
                    try:
                        # Parse received data without logging raw message
                        message_data = json.loads(message)

                        # Process the message
                        await process_websocket_game(message_data)

                        # In test mode, process only a few games
                        if args.test:
                            count = await count_games_since_last_run()
                            if count >= 10:
                                console.print(
                                    "[cyan]Test mode complete[/cyan]")
                                return
                    except json.JSONDecodeError:
                        print_error(f"Invalid JSON from WebSocket")
                    except Exception as e:
                        print_error(f"Error processing message: {str(e)}")
        except websockets.ConnectionClosed as e:
            # Log connection closed with nicer format
            log_websocket_connection("reconnecting")
            await asyncio.sleep(reconnect_delay)
            # Exponential backoff for reconnection attempts, max 60 seconds
            reconnect_delay = min(reconnect_delay * 1.5, 60)
        except websockets.WebSocketException as e:
            # Log connection error with nicer format
            log_websocket_connection("reconnecting")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, 60)
        except Exception as e:
            print_error(f"Unexpected error: {str(e)}")
            log_websocket_connection("reconnecting")
            await asyncio.sleep(reconnect_delay)


async def count_games_since_last_run() -> int:
    """
    Count the number of games processed since the start of this run.
    Used for test mode to limit processing.
    """
    db = get_db()
    try:
        # Get initial game count when the script started
        count = db.query(Game).count()
        return count
    finally:
        db.close()


async def quick_catchup(last_game_id: Optional[str] = None) -> bool:
    """
    Perform a quick catchup to process any games missed since last run.

    Args:
        last_game_id: Last processed game ID

    Returns:
        Boolean indicating if any new streaks were processed
    """
    logger.debug(
        f"Starting quick catchup from game ID {last_game_id or 'beginning'}")

    # Get max streak number from database to ensure unique numbering
    db = get_db()
    try:
        max_streak_number = db.query(
            func.max(Streak.streak_number)).scalar() or 0
        logger.debug(f"Highest streak number in database: {max_streak_number}")
    finally:
        db.close()

    # Determine how many games to fetch for catchup
    db = get_db()
    try:
        # Get max game ID from database
        max_game_id = db.query(func.max(Game.game_id)).scalar()
        if not max_game_id:
            logger.debug("No games in database, nothing to catch up")
            return False

        if not last_game_id:
            # If no last_game_id, start from the earliest game
            min_game_id = db.query(func.min(Game.game_id)).scalar()
            # Start right before the first game
            last_game_id = str(int(min_game_id) - 1)

        # Calculate how many games we need to process
        games_to_process = db.query(Game).filter(
            Game.game_id > last_game_id).count()

        if games_to_process > 0:
            print_info(f"Catching up {games_to_process} games")
    finally:
        db.close()

    if not games_to_process:
        return False

    # Process all missed games at once
    processed_streaks = 0
    # Start numbering after the highest existing streak
    current_streak_offset = max_streak_number

    # Fetch all missed games at once - no need for batch size
    games_df = fetch_latest_games(last_game_id)

    if games_df.empty:
        return False

    # Process all games
    streaks_df = process_streaks(games_df)

    # Adjust streak numbers to avoid conflicts with existing streaks
    if not streaks_df.empty:
        # Adjust streak numbers to continue from the highest number in the database
        original_numbers = streaks_df['streak_number'].tolist()
        streaks_df['streak_number'] = streaks_df['streak_number'] + \
            current_streak_offset
        new_numbers = streaks_df['streak_number'].tolist()
        logger.debug(
            f"Adjusted streak numbers from {original_numbers} to {new_numbers}")

        # Save streaks
        saved_ids = save_streaks(streaks_df)
        processed_streaks = len(saved_ids)
        if processed_streaks > 0:
            print_success(
                f"Processed {processed_streaks} new streaks during catchup")

            # Get all streaks for prediction in a unified process
            all_streaks_df = get_all_streaks()

            # Make prediction for the next streak and process in a single message
            prediction = make_prediction(all_streaks_df)

            # Save prediction if valid
            if prediction and 'next_streak_number' in prediction:
                db = get_db()
                try:
                    save_prediction(max_streak_number +
                                    len(streaks_df), prediction, db)

                    # Update correctness of previous predictions
                    updated_count = update_prediction_correctness()
                    if updated_count > 0:
                        print_info(
                            f"Updated {updated_count} previous prediction(s) with actual results")
                finally:
                    db.close()

    return processed_streaks > 0


def continuous_processing() -> None:
    """Run continuous processing looking for new games.
    Performs a small catchup first to handle any missed streaks,
    then continuously monitors for new games."""
    print_info("Starting continuous processing")

    # Get last processed game ID and streak info
    current_streak_number, last_game_id = get_streak_info()

    if current_streak_number:
        print_info(
            f"Starting from streak #{current_streak_number} (last game ID: {last_game_id})")
    else:
        print_info("No existing streaks found - will start from the beginning")

    # First do a quick catchup for any missed games
    catchup_completed = quick_catchup(last_game_id)

    # Get updated last processed game ID
    _, last_game_id = get_streak_info()
    logger.debug(
        f"After catchup phase, last processed game ID: {last_game_id}")

    # Run continuous processing
    while True:
        try:
            # Get current max streak number for numbering new streaks
            db = get_db()
            try:
                max_streak_number = db.query(
                    func.max(Streak.streak_number)).scalar() or 0
                logger.debug(
                    f"Current highest streak number: {max_streak_number}")
            finally:
                db.close()

            # 1. Fetch latest games
            games_df = fetch_latest_games(last_game_id)

            if not games_df.empty:
                # Update last game ID
                last_game_id = str(games_df['game_id'].max())
                logger.debug(
                    f"Fetched {len(games_df)} new games up to game ID {last_game_id}")

                # 2. Process streaks
                streaks_df = process_streaks(games_df)

                if not streaks_df.empty:
                    # Adjust streak numbers to avoid conflicts
                    original_numbers = streaks_df['streak_number'].tolist()
                    streaks_df['streak_number'] = streaks_df['streak_number'] + \
                        max_streak_number
                    new_numbers = streaks_df['streak_number'].tolist()
                    logger.debug(
                        f"Adjusted streak numbers from {original_numbers} to {new_numbers}")

                    # 3. Save streaks
                    saved_ids = save_streaks(streaks_df)
                    if saved_ids:
                        logger.debug(f"Saved {len(saved_ids)} new streaks")

                        # 4. Get all streaks for prediction
                        all_streaks_df = get_all_streaks()

                        # 5. Make prediction for the next streak and handle in one workflow
                        prediction = make_prediction(all_streaks_df)

                        # 6. Save prediction if valid
                        if prediction and 'next_streak_number' in prediction:
                            db = get_db()
                            try:
                                save_prediction(
                                    max_streak_number + len(streaks_df), prediction, db)

                                # 7. Update correctness of previous predictions
                                updated_count = update_prediction_correctness()
                                if updated_count > 0:
                                    print_info(
                                        f"Updated {updated_count} previous prediction(s) with actual results")
                            finally:
                                db.close()
                else:
                    logger.debug("No new streaks found in this batch")
            else:
                logger.debug("No new games found, waiting for next check")

        except Exception as e:
            print_error(f"Error in processing cycle: {str(e)}")

        # In test mode, process only once
        if args.test:
            print_info("Test mode - stopping after one cycle")
            break

        # Sleep before next iteration
        logger.debug(f"Sleeping for {FETCH_INTERVAL} seconds")
        time.sleep(FETCH_INTERVAL)


def main():
    """Main function."""
    # Use WebSocket or polling based on args
    if args.websocket:
        # Run WebSocket-based continuous processing
        try:
            asyncio.run(websocket_continuous_processing())
        except KeyboardInterrupt:
            print_info("Shutting down...")
        except Exception as e:
            print_error(f"Error in WebSocket processing: {str(e)}")
    else:
        # Run traditional polling-based continuous processing
        continuous_processing()


if __name__ == "__main__":
    main()
