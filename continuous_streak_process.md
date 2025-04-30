# Continuous Streak Processing

The project includes a continuous streak processor that runs as a standalone service. This processor:

1. Fetches new crash game data from the database
2. Processes data into 10× streaks
3. Makes predictions for upcoming streaks
4. Updates previous predictions with correctness results

## Running the Processor

You can run the processor locally:

```bash
source crash_env/bin/activate
python api/streak_processor.py
```

## Deployment on Railway

The project includes a `railway.toml` file for easy deployment on Railway:

1. Install the Railway CLI: `npm i -g @railway/cli`
2. Login to Railway: `railway login`
3. Link your project: `railway link`
4. Deploy: `railway up`

## Environment Variables

The streak processor uses these environment variables:

- `DATABASE_URL`: PostgreSQL database connection string
- `MULTIPLIER_THRESHOLD`: Threshold for streak detection (default: 10.0)
- `FETCH_INTERVAL`: Seconds between processing cycles (default: 60)
- `MODEL_PATH`: Path to the trained model file (default: ../output/temporal_model.pkl)
- `LOG_LEVEL`: Logging level (default: INFO)

## Logs

Logs are written to `api/logs/streak_processor.log` and also output to the console.
