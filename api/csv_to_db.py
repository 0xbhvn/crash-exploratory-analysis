#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for importing data from CSV files into the database.
Extracts streaks from games CSV and supports bulk importing of streaks and predictions.
"""

import os
import logging
import pandas as pd
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs/csv_to_db.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Constants
MULTIPLIER_THRESHOLD = float(os.getenv("MULTIPLIER_THRESHOLD", "10.0"))

# Import necessary modules
try:
    from app.database.db_config import Base
    from app.models.streak import Streak
    from app.models.prediction import Prediction
    from app.models.game import Game
    from data_processing import extract_streaks_and_multipliers
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {str(e)}")
    raise

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

logger.info(
    f"Connecting to database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'DB'}")

try:
    engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
    # Test the connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        if result.scalar() == 1:
            logger.info("Database connection test successful")
        else:
            logger.error("Database connection test failed")
except Exception as e:
    logger.error(f"Error connecting to database: {str(e)}")
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        logger.error(f"Error creating database session: {str(e)}")
        raise


def reset_database_sequences():
    """Reset database sequences to start IDs from 1."""
    logger.info("Resetting database sequences...")
    db = get_db()
    try:
        # Check if there are any existing streaks
        streak_count = db.query(Streak).count()
        if streak_count > 0:
            logger.warning(
                f"Database already contains {streak_count} streaks. Cannot reset sequences safely.")
            return False

        # Reset the sequence for streaks table
        db.execute(text("ALTER SEQUENCE streaks_id_seq RESTART WITH 1"))
        db.commit()
        logger.info("Reset streaks_id_seq to start from 1")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting database sequences: {str(e)}")
        return False
    finally:
        db.close()


def catchup_processing_with_csv() -> None:
    """Process all historical games to extract streaks only.
    Uses games.csv as input and creates a streaks CSV for bulk importing.
    Predictions will only be generated in live processing mode."""
    logger.info("Starting historical catchup processing with CSV (streaks only)")

    # Create temporary directory for CSV files if it doesn't exist
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    # Input and output files
    games_csv = os.path.join(project_root, "games.csv")
    streaks_csv = os.path.join(temp_dir, "streaks.csv")

    # Check if games.csv exists
    if not os.path.exists(games_csv):
        logger.error(f"Games CSV file not found at {games_csv}")
        return

    logger.info(f"Using games CSV file: {games_csv}")

    # Reset database sequences if needed to start IDs from 1
    reset_success = reset_database_sequences()
    if not reset_success:
        logger.warning(
            "Database sequences were not reset. IDs may not start from 1.")

    # Get existing streaks to avoid duplicates
    db = get_db()
    try:
        max_streak_number = db.query(
            func.max(Streak.streak_number)).scalar() or 0
        existing_streak_numbers = set(
            r[0] for r in db.query(Streak.streak_number).all())
        logger.info(
            f"Found {max_streak_number} as highest streak number in database")
        logger.info(
            f"Database contains {len(existing_streak_numbers)} existing streaks")
    finally:
        db.close()

    # Step 1: Load games.csv and extract all streaks in one pass
    logger.info("Loading games.csv and extracting streaks...")

    try:
        # Load games data
        games_df = pd.read_csv(games_csv)
        logger.info(f"Loaded {len(games_df)} games from CSV")

        # Rename columns to match expected format if needed
        if 'Game ID' not in games_df.columns and 'game_id' in games_df.columns:
            games_df = games_df.rename(columns={'game_id': 'Game ID'})
        if 'Bust' not in games_df.columns and 'crash_point' in games_df.columns:
            games_df = games_df.rename(columns={'crash_point': 'Bust'})

        # Extract all streaks
        streaks_df = extract_streaks_and_multipliers(
            games_df, MULTIPLIER_THRESHOLD)
        logger.info(f"Extracted {len(streaks_df)} streaks from games data")

        # Check for streaks that already exist in the database
        original_count = len(streaks_df)
        streaks_df = streaks_df[~streaks_df['streak_number'].isin(
            existing_streak_numbers)]
        filtered_count = original_count - len(streaks_df)
        logger.info(
            f"Filtered out {filtered_count} streaks that already exist in the database")

        # If there are new streaks, add them with proper numbering
        if len(streaks_df) > 0:
            # Start new streak numbers after the highest existing streak number
            next_streak_number = max_streak_number + 1
            logger.info(
                f"Will add {len(streaks_df)} new streaks starting from streak number {next_streak_number}")

            # Renumber the streaks to continue from max_streak_number
            streaks_df['streak_number'] = range(
                next_streak_number, next_streak_number + len(streaks_df))

        # Only keep columns that match the Streak model
        streak_columns = [
            'streak_number',
            'start_game_id',
            'end_game_id',
            'streak_length',
            'hit_multiplier',
            'mean_multiplier',
            'std_multiplier',
            'max_multiplier',
            'min_multiplier'
        ]

        # Ensure all required columns exist
        for col in streak_columns:
            if col not in streaks_df.columns:
                if col == 'std_multiplier':  # This one is nullable
                    streaks_df[col] = None
                else:
                    logger.error(
                        f"Required column {col} not found in streaks DataFrame")
                    return

        # If there are no new streaks, we're done
        if streaks_df.empty:
            logger.info("No new streaks to save to database")
            return

        # Save only the relevant columns to CSV
        streaks_df[streak_columns].to_csv(streaks_csv, index=False)
        logger.info(f"Saved {len(streaks_df)} new streaks to {streaks_csv}")

        # Bulk insert streaks from CSV
        streak_count = bulk_insert_streaks_from_csv(streaks_csv)
        logger.info(f"Inserted {streak_count} streaks into database")

    except Exception as e:
        logger.error(f"Error in catchup processing: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(streaks_csv):
                os.remove(streaks_csv)
            logger.info("Cleaned up temporary CSV files")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {str(e)}")

    logger.info(
        "Catchup complete - predictions will be generated in live processing mode")


def bulk_insert_streaks_from_csv(csv_path: str, batch_size: int = 5000) -> int:
    """
    Bulk insert streaks from a CSV file into the database.

    Args:
        csv_path: Path to CSV file with streak data
        batch_size: Number of records to insert in each batch

    Returns:
        Number of streaks inserted
    """
    logger.info(
        f"Bulk inserting streaks from {csv_path} (batch size: {batch_size})")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0

        # Convert DataFrame to list of dictionaries
        streak_dicts = df.to_dict(orient='records')
        inserted_count = 0

        # Get existing streak numbers to avoid duplicates
        db = get_db()
        try:
            existing_streak_numbers = set(
                r[0] for r in db.query(Streak.streak_number).all())
        finally:
            db.close()

        # Filter out streaks that already exist
        new_streaks = [s for s in streak_dicts if int(
            s['streak_number']) not in existing_streak_numbers]
        logger.info(
            f"Found {len(new_streaks)} new streaks out of {len(streak_dicts)} total")

        # Process in batches
        for i in range(0, len(new_streaks), batch_size):
            batch = new_streaks[i:i+batch_size]
            db = get_db()
            try:
                # Convert to Streak objects
                streak_objects = []
                for s in batch:
                    streak = Streak(
                        streak_number=int(s['streak_number']),
                        start_game_id=int(s['start_game_id']),
                        end_game_id=int(s['end_game_id']),
                        streak_length=int(s['streak_length']),
                        hit_multiplier=float(s['hit_multiplier']),
                        mean_multiplier=float(s['mean_multiplier']),
                        std_multiplier=float(s['std_multiplier']) if 'std_multiplier' in s and pd.notna(
                            s['std_multiplier']) else None,
                        max_multiplier=float(s['max_multiplier']),
                        min_multiplier=float(s['min_multiplier'])
                    )
                    streak_objects.append(streak)

                # Bulk insert
                db.bulk_save_objects(streak_objects)
                db.commit()
                inserted_count += len(batch)
                logger.info(
                    f"Inserted batch of {len(batch)} streaks (total: {inserted_count})")
            except Exception as e:
                db.rollback()
                logger.error(f"Error inserting batch: {str(e)}")
                raise
            finally:
                db.close()

        return inserted_count
    except Exception as e:
        logger.error(f"Error in bulk_insert_streaks_from_csv: {str(e)}")
        raise


def bulk_insert_predictions_from_csv(csv_path: str, batch_size: int = 5000) -> int:
    """
    Bulk insert predictions from a CSV file into the database.

    Args:
        csv_path: Path to CSV file with prediction data
        batch_size: Number of records to insert in each batch

    Returns:
        Number of predictions inserted
    """
    logger.info(
        f"Bulk inserting predictions from {csv_path} (batch size: {batch_size})")

    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        if df.empty:
            return 0

        # Convert DataFrame to list of dictionaries
        prediction_dicts = df.to_dict(orient='records')

        # Get existing predictions to avoid duplicates
        db = get_db()
        try:
            existing_streak_numbers = set(r[0] for r in db.query(
                Prediction.next_streak_number).all())
        finally:
            db.close()

        # Filter out predictions that already exist
        new_predictions = [p for p in prediction_dicts if int(
            p['next_streak_number']) not in existing_streak_numbers]
        logger.info(
            f"Found {len(new_predictions)} new predictions out of {len(prediction_dicts)} total")

        inserted_count = 0

        # Process in batches
        for i in range(0, len(new_predictions), batch_size):
            batch = new_predictions[i:i+batch_size]
            db = get_db()
            try:
                # Convert to Prediction objects
                pred_objects = []
                for p in batch:
                    # Convert prediction_data from string if needed
                    if isinstance(p['prediction_data'], str):
                        import json
                        p['prediction_data'] = json.loads(p['prediction_data'])

                    pred = Prediction(
                        next_streak_number=int(p['next_streak_number']),
                        starts_after_game_id=int(p['starts_after_game_id']),
                        predicted_cluster=int(p['predicted_cluster']),
                        prediction_desc=str(p['prediction_desc']),
                        confidence=float(p['confidence']),
                        prediction_data=p['prediction_data'],
                        correct=p['correct']
                    )
                    pred_objects.append(pred)

                # Bulk insert
                db.bulk_save_objects(pred_objects)
                db.commit()
                inserted_count += len(batch)
                logger.info(
                    f"Inserted batch of {len(batch)} predictions (total: {inserted_count})")
            except Exception as e:
                db.rollback()
                logger.error(f"Error inserting prediction batch: {str(e)}")
                raise
            finally:
                db.close()

        return inserted_count
    except Exception as e:
        logger.error(f"Error in bulk_insert_predictions_from_csv: {str(e)}")
        raise


if __name__ == "__main__":
    # This module can be run directly to perform CSV import
    logger.info("CSV to DB utility starting")
    catchup_processing_with_csv()
