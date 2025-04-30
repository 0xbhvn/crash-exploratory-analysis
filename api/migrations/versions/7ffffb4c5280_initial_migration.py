"""Initial migration

Revision ID: 7ffffb4c5280
Revises: 
Create Date: 2023-04-30 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ffffb4c5280'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This is just a placeholder since the tables already exist
    pass


def downgrade() -> None:
    # This is just a placeholder
    pass
