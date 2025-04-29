# Crash Game Streak Analysis

A tool for analyzing and predicting streak lengths before configurable multipliers in "crash"-style gambling games.

## Overview

This project analyzes game data from "crash"-style gambling games, focusing on streaks before a configurable multiplier threshold (default: 10×). It builds a machine learning model to predict the length category of streaks before the next multiplier at or above the threshold occurs.

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

# Run analysis with a different multiplier threshold (e.g. 2×)
python main.py --input games.csv --multiplier_threshold 2.0 --save_plots

# Run analysis with custom percentile boundaries for clustering
python main.py --input games.csv --percentiles 0.33,0.66 --save_plots
```

This will:

1. (Optional) Update data from the database
2. Load and clean the game data
3. Analyze streak lengths
4. Generate visualizations (if `--save_plots` is specified)
5. Prepare features for machine learning
6. Train a model to predict streak length clusters
7. Save results to the specified output directory

### Percentile-Based Clustering

The analysis now supports customizable percentile boundaries for clustering streak lengths. By default, the system uses quartiles (25%, 50%, 75%), creating 4 clusters:

- Cluster 0: Bottom 25% (shortest streaks)
- Cluster 1: 25-50 percentile
- Cluster 2: 50-75 percentile
- Cluster 3: Top 25% (longest streaks)

You can customize these boundaries using the `--percentiles` parameter with comma-separated values:

```bash
# Create 3 clusters using tertiles (0-33%, 33-66%, 66-100%)
python main.py --percentiles 0.33,0.66

# Create 5 clusters with custom boundaries
python main.py --percentiles 0.20,0.40,0.60,0.80
```

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

### Temporal Analysis

To analyze temporal patterns:

```bash
python temporal_analysis.py \
  --mode train \
  --input games.csv \
  --use_class_weights --weight_scale 1.25 \
  --max_depth 8 \
  --eta 0.03 \
  --num_rounds 1200 --early_stopping 150 \
  --gamma 1 --reg_lambda 1.2 --min_child_weight 3 \
  --subsample 0.8 --colsample_bytree 0.8
```

This will:

1. Train a model on the training data
2. Evaluate the model on the test set
3. Save the model and evaluation results

### Predicting Streak Lengths

To predict streak lengths:

```bash
python temporal_analysis.py --mode predict --input games.csv --model_path xgb_model.pkl
```

This will:

1. Load the trained model
2. Make predictions on the input data
3. Save the predictions

## Command Line Arguments

- `--input`: Path to input CSV file with Game ID and Bust columns (default: games.csv)
- `--multiplier_threshold`: Threshold for considering a multiplier as a hit (default: 10.0)
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
- `--percentiles`: Comma-separated list of percentile boundaries for clustering (default: 0.25,0.50,0.75)

## Outputs

The analysis generates several outputs in the specified output directory:

- `streak_lengths.csv`: Raw streak lengths before threshold multipliers
- `streak_histogram.png`: Histogram of streak lengths
- `streak_percentiles.png`: Histogram with percentile markers
- `feature_importance.png`: Feature importance plot from the trained model
- `calibration_curve.png`: Model calibration curve
- `confusion_matrix.csv`: Confusion matrix for test set predictions
- `xgboost_model.pkl`: Serialized trained model

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Recent Updates

### New True Prediction Mode

A new `true_predict` mode has been added to the temporal analysis that enforces strict temporal boundaries:

```bash
python -m temporal.app --mode true_predict --input games.csv --output_dir ./output --multiplier_threshold 10.0 --num_streaks 200
```

This mode ensures:

- Only past data is used for each prediction
- No data leakage occurs by enforcing strict boundaries
- Predictions more accurately reflect real-world performance

### Comparing Prediction Methods

The project now offers three different prediction approaches:

1. **Standard Temporal Prediction**: Evaluates on historical data

   ```bash
   python -m temporal.app --mode predict --input games.csv --output_dir ./output
   ```

   - Shows artificially high confidence for some points (up to 100%)
   - Useful for testing model performance on historical data

2. **True Temporal Prediction**: Strictly forward-looking

   ```bash
   python -m temporal.app --mode true_predict --input games.csv --output_dir ./output
   ```

   - Realistic confidence scores (typically 40-60%)
   - Prevents any data leakage from future observations
   - More representative of actual predictive power

3. **Balanced Prediction**: Uses ensemble approach

   ```bash
   python predict_only.py
   ```

   - Combines multiple prediction methods (model, historical, transitions)
   - Provides more balanced probability estimates
   - Most robust for actual prediction use

## Prediction Accuracy

True prediction mode shows an overall accuracy of around 48%, with class-specific accuracies:

- Short streaks (1-3): ~58% accuracy
- Medium-short streaks (4-7): ~39% accuracy
- Medium-long streaks (8-14): ~47% accuracy
- Long streaks (>14): ~46% accuracy

## Usage with True Prediction Mode

To train a new model:

```bash
python -m temporal.app --mode train --input games.csv --output_dir ./output
```

To make predictions about future streaks:

```bash
python predict_only.py
```

## Project Structure with True Prediction Mode

- [main.py](mdc:main.py): Main application entry point
- [analyzer.py](mdc:analyzer.py): Core analysis functionality
- [data_processing.py](mdc:data_processing.py): Data preparation
- [modeling.py](mdc:modeling.py): Model training and evaluation
- [visualization.py](mdc:visualization.py): Visualizations
- [predict_only.py](mdc:predict_only.py): Balanced prediction script
- [temporal/](mdc:temporal/): Temporal analysis module
  - [app.py](mdc:temporal/app.py): Command-line interface
  - [true_predict.py](mdc:temporal/true_predict.py): True prediction mode
  - [features.py](mdc:temporal/features.py): Temporal feature engineering
  - [prediction.py](mdc:temporal/prediction.py): Prediction functionality
