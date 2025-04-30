#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reset the alembic_version table to the initial migration.
"""

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

# Load environment variables
load_dotenv()

# Connect to the database
engine = create_engine(os.getenv('DATABASE_URL'))

# Update the alembic_version table
with engine.connect() as conn:
    conn.execute(
        text("UPDATE alembic_version SET version_num = '7ffffb4c5280'"))
    conn.commit()
    print("Successfully reset alembic_version to initial migration (7ffffb4c5280)")
