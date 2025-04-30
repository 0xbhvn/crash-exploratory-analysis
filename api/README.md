# Crash Game Streak Prediction API

This API service provides endpoints for predicting 10× streaks in crash games based on temporal patterns.

## Features

- **FastAPI Backend**: High-performance asynchronous API with automatic documentation
- **Streaming WebSocket**: Real-time game data processing and predictions
- **Temporal Model Integration**: Uses a trained XGBoost model to predict streak lengths
- **Database Integration**: Stores games, streaks, and predictions
- **Notification System**: Sends alerts for high-confidence predictions
- **Dockerized Deployment**: Easy deployment with Docker Compose
- **Migration Support**: Database migration using Alembic

## Architecture

The system consists of the following components:

1. **API Service**: FastAPI application exposing endpoints
2. **Database**: PostgreSQL for storing games, streaks, and predictions
3. **Collector Service**: Service that polls for new data and generates predictions
4. **WebSocket Connection**: Real-time data streaming from games to the API

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- PostgreSQL (if running without Docker)

### Environment Setup

Create a `.env` file in the root directory with the following variables:

```text
# Database
DATABASE_URL=postgresql://postgres:postgres@db:5432/crash_game
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=crash_game

# API
API_URL=http://api:8000
MULTIPLIER_THRESHOLD=10.0
PREDICTION_CONFIDENCE_THRESHOLD=0.75
COLLECTOR_SLEEP_SECONDS=30

# Notification (optional)
SLACK_WEBHOOK=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```

### Running with Docker

Build and start all services:

```bash
cd api
docker-compose up -d
```

### Running without Docker

1. Install dependencies:

    ```bash
    source crash_env/bin/activate
    pip install -r requirements.txt
    ```

2. Run database migrations:

    ```bash
    cd api
    alembic upgrade head
    ```

3. Start the API server:

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```

4. In a separate terminal, run the collector:

    ```bash
    python -m app.collector
    ```

## API Endpoints

- `GET /`: Health check
- `GET /docs`: Interactive API documentation
- `WebSocket /ws`: WebSocket endpoint for real-time updates
- `POST /predictions/`: Make a prediction for the next streak
- `GET /predictions/`: Get recent predictions
- `GET /streaks/`: Get recent streaks
- `GET /games/`: Get recent games

## Collector Service

The collector service runs in a separate container and:

1. Fetches new game data incrementally
2. Processes data into streaks
3. Makes predictions when a 10× streak is detected
4. Sends notifications for high-confidence predictions

## Database Migrations

To create a new migration:

```bash
cd api
alembic revision --autogenerate -m "Description of changes"
```

To apply migrations:

```bash
alembic upgrade head
```

## Development

To set up the development environment:

1. Create a virtual environment:

    ```bash
    python -m venv crash_env
    source crash_env/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the development server:

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
