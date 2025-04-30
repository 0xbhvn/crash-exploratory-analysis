#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Database configuration for the API service.
Handles connection to the database and provides session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment variables")

# Create optimized engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,       # Maintain 10 connections
    max_overflow=20,    # Allow up to 20 additional connections
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_timeout=30     # Wait up to 30 seconds for a connection
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for declarative models
Base = declarative_base()

# Define function to get database session


def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # We don't close the DB here anymore - caller is responsible for closing
