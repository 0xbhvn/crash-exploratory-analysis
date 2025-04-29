# Temporal Analysis for Crash Game 10× Streak Prediction

This package provides temporal analysis and prediction for crash game streaks.

## Key Features

- Strict temporal separation to eliminate data leakage
- Purely historical features (no current streak properties)
- Focus on genuinely predictive temporal patterns
- Realistic evaluation metrics for time-series prediction
- Transition analysis to measure pattern prediction power
- Forward-looking prediction for next streaks

## Usage

The package supports different modes of operation:

### Train Mode

Train a new model on historical data:

```bash
python -m temporal.app --mode train --input games.csv --output_dir ./output
```

### Predict Mode

Make predictions on historical data:

```bash
python -m temporal.app --mode predict --input games.csv --output_dir ./output
```

### True Predict Mode

Make strict forward-looking predictions with no data leakage:

```bash
python -m temporal.app --mode true_predict --input games.csv --output_dir ./output
```

### Next Streak Mode

Predict the next streak after the latest available data:

```bash
python -m temporal.app --mode next_streak --input games.csv --output_dir ./output
```

## Forward Deployment

The package includes functionality to deploy the model for predicting the next streak after the latest available data. This can be used in a production environment to make real-time predictions.

### Using the Deployment API

```python
from temporal.deploy import setup_prediction_service, predict_next_streak
import pandas as pd

# Load your streak data
streak_df = pd.read_csv('games.csv')

# Set up the prediction service
model_bundle = setup_prediction_service(model_dir='./output')

# Predict the next streak
prediction = predict_next_streak(model_bundle, streak_df)

# Access prediction results
predicted_cluster = prediction['predicted_cluster']
prediction_desc = prediction['prediction_desc']
confidence = prediction['confidence']
```

### Example Script

An example script is provided in `temporal/examples/next_streak_prediction.py`:

```bash
python -m temporal.examples.next_streak_prediction --input games.csv --model_dir ./output
```

## Module Structure

- `loader.py`: Data loading functionality
- `features.py`: Feature engineering
- `splitting.py`: Temporal train-test splitting
- `training.py`: Model training
- `evaluation.py`: Performance evaluation
- `prediction.py`: General prediction functionality
- `true_predict.py`: Strict temporal prediction
- `deploy.py`: Deployment functionality for next streak prediction
- `app.py`: Command-line interface 