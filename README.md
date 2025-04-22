# Crash Game 10× Streak Analysis

A tool for analyzing and predicting streak lengths before 10× multipliers in "crash"-style gambling games.

## Overview

This project analyzes game data from "crash"-style gambling games, focusing on streaks before 10× multipliers. It builds a machine learning model to predict the length category of streaks before the next 10× multiplier occurs.

## Project Structure

The codebase has been refactored into a modular structure:

- `main.py` - Entry point script with command-line interface
- `analyzer.py` - Core `CrashStreakAnalyzer` class that coordinates the analysis
- `data_processing.py` - Functions for data loading, cleaning, and feature engineering
- `modeling.py` - Model training, evaluation, and prediction functions
- `visualization.py` - Plotting and visualization functions
- `daily_updates.py` - Handling daily updates and drift detection
- `fetch_data.py` - Script to fetch data from the database
- `crash_streak_analysis.py` - Wrapper script for backward compatibility

## Requirements

- Python 3.10+
- pandas
- numpy
- matplotlib
- xgboost
- scikit-learn
- joblib
- sqlalchemy (for database connectivity)
- python-dotenv (for environment variable management)

## Installation

```bash
python -m venv crash_env
source crash_env/bin/activate  # On Windows: .\crash_env\Scripts\activate
pip install -r requirements.txt
```

## Database Configuration

Create a `.env` file in the project root with your database credentials:

```text
# Copy from env.template to .env and update with your credentials
DATABASE_URL=postgresql://username:password@hostname:port/database
```

The application will automatically load the connection details from this file, so you don't need to set environment variables manually.

## Usage

### Data Fetching

Fetch data from the database:

```bash
# Fetch all data
python fetch_data.py --output games.csv

# Fetch only new data since last known game_id
python fetch_data.py --incremental --output games.csv

# Limit the number of rows fetched
python fetch_data.py --limit 10000 --output games.csv
```

### Basic Analysis

```bash
# Run analysis with automatic data update from database
python main.py --update_csv --save_plots

# Run analysis with existing data file
python main.py --input games.csv --output_dir ./results --save_plots
```

This will:

1. (Optional) Update data from the database
2. Load and clean the game data
3. Analyze streak lengths
4. Generate visualizations (if `--save_plots` is specified)
5. Prepare features for machine learning
6. Train a model to predict streak length clusters
7. Save results to the specified output directory

### Daily Updates

To update the model with new data:

```bash
python main.py --input games.csv --update new_games.csv --drift_threshold 0.005
```

This will:

1. Load the existing data
2. Add new data
3. Check for distribution drift
4. Retrain the model if drift is detected

## Command Line Arguments

- `--input`: Path to input CSV file with Game ID and Bust columns (default: games.csv)
- `--window`: Rolling window size for feature engineering (default: 50)
- `--test_frac`: Fraction of data to use for testing (default: 0.2)
- `--output_dir`: Directory to save outputs (default: ./output)
- `--random_seed`: Random seed for reproducibility (default: 42)
- `--save_plots`: Generate and save plots (flag)
- `--update`: Path to new data file for model update
- `--drift_threshold`: Threshold for detecting distribution drift (default: 0.005)
- `--update_csv`: Update the CSV data from the database before analysis
- `--full_fetch`: Fetch all data instead of just new records (with --update_csv)
- `--fetch_limit`: Limit the number of rows to fetch from database

## Outputs

The analysis generates several outputs in the specified output directory:

- `streak_lengths.csv`: Raw streak lengths before 10× multipliers
- `streak_histogram.png`: Histogram of streak lengths
- `streak_percentiles.png`: Histogram with percentile markers
- `feature_importance.png`: Feature importance plot from the trained model
- `calibration_curve.png`: Model calibration curve
- `confusion_matrix.csv`: Confusion matrix for test set predictions
- `xgboost_model.pkl`: Serialized trained model

## License

This project is licensed under the MIT License - see the LICENSE file for details.
