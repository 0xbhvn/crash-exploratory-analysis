#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Check the alembic_version table in the database.
"""

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

# Load environment variables
load_dotenv()

# Connect to the database
engine = create_engine(os.getenv('DATABASE_URL'))

# Query the alembic_version table
with engine.connect() as conn:
    try:
        result = conn.execute(text('SELECT version_num FROM alembic_version'))
        versions = [row[0] for row in result]
        print(f"Current Alembic versions: {versions}")
    except Exception as e:
        print(f"Error querying alembic_version table: {str(e)}")
