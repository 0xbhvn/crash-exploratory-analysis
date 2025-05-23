"""remove_prediction_data_and_day_of_week_fields

Revision ID: dc5f94f49c90
Revises: cfe65930d537
Create Date: 2025-04-30 11:09:25.328931

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'dc5f94f49c90'
down_revision: Union[str, None] = 'cfe65930d537'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('streak_predictions', 'day_of_week')
    op.drop_column('streak_predictions', 'prediction_data')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('streak_predictions', sa.Column('prediction_data', postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=False))
    op.add_column('streak_predictions', sa.Column('day_of_week', sa.INTEGER(), autoincrement=False, nullable=True))
    # ### end Alembic commands ### 