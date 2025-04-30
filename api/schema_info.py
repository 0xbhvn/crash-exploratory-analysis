#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to print database schema information.
"""

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, MetaData
import os

# Load environment variables
load_dotenv()

# Connect to the database
engine = create_engine(os.getenv('DATABASE_URL'))
inspector = inspect(engine)

# Get all table names
table_names = inspector.get_table_names()
print(f"Found {len(table_names)} tables: {', '.join(table_names)}")

# Print schema for each table
for table_name in table_names:
    print(f"\n{table_name} columns:")
    for column in inspector.get_columns(table_name):
        print(
            f"  - {column['name']}: {column['type']} (nullable={column['nullable']})")

    # Print primary keys
    pk = inspector.get_pk_constraint(table_name)
    print(f"  Primary keys: {pk['constrained_columns']}")

    # Print foreign keys
    foreign_keys = inspector.get_foreign_keys(table_name)
    if foreign_keys:
        print("  Foreign keys:")
        for fk in foreign_keys:
            print(
                f"    - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")

    # Print indexes
    indexes = inspector.get_indexes(table_name)
    if indexes:
        print("  Indexes:")
        for idx in indexes:
            print(
                f"    - {idx['name']}: {idx['column_names']} (unique={idx.get('unique', False)})")
