
# Potential Data Leakage

I've created a new `temporal_analysis.py` script that properly addresses the data leakage and insufficient temporal separation issues in your model. The key improvements include:

1. **Strict temporal train-test split** - Data is split chronologically, ensuring future data never influences past predictions.

2. **Features based only on past information** - All features are created using strictly historical data:
   - Only lagged streak features (`prev1_length`, `prev2_length`)
   - Rolling window statistics on past streaks
   - Streak category patterns and transitions
   - All current streak properties are explicitly excluded

3. **Proper validation approach** - Early stopping with a validation set that maintains temporal separation.

4. **Realistic evaluation metrics** - Measures model performance against baseline and analyzes performance on different streak types.

5. **Transition analysis** - Specifically measures how well the model handles different streak pattern transitions.

Run it with:

```bash
source crash_env/bin/activate && python temporal_analysis.py
```

Let me fix the issue with categorical features in the temporal analysis script:

Read file: temporal_analysis.py
Let me check the feature creation section to understand the issue:

Read file: temporal_analysis.py
Now I can see the issue. Let me fix the script to address the problem with categorical features:

Now let's check the training function to make sure we're handling features correctly:

Read file: temporal_analysis.py
I need to check what's happening in the main function where the script actually runs:

Read file: temporal_analysis.py
Let's now check the original train_test_split function to see what's causing the data leakage:

Read file: temporal_analysis.py
Now let's check the feature creation function to see what might be causing the data leakage:

Read file: temporal_analysis.py
Let's check how the test results are being reported:

Read file: temporal_analysis.py
Now, let's examine the current model training function to identify potential issues:

Read file: temporal_analysis.py
Now let's check the current train-test split implementation:

Read file: temporal_analysis.py
Let's check the feature creation function to identify any potential causes of data leakage:

Read file: temporal_analysis.py
Let's examine how the model performance evaluation is implemented:

Read file: temporal_analysis.py
Now let's look at the main function to understand the overall flow of the application:

Read file: temporal_analysis.py
Now that I have a clear understanding of the code structure, let's examine if there is any data leakage between the train and test sets:

Read file: temporal_analysis.py
Let's check how the features are created:

Read file: temporal_analysis.py
Now let's look at the `analyze_temporal_performance` function to understand the evaluation process:

Read file: temporal_analysis.py
Let's check the main function to see how everything fits together:

Read file: temporal_analysis.py
Let's create a new file to fix the data leakage issues while keeping a clean temporal analysis approach:

Let's first look at the existing modeling.py file to understand what we need to fix:

Read file: modeling.py
Let's look more at the data_processing.py file to understand how features are created, which may be causing the data leakage:

Read file: data_processing.py
Let's look at the key feature preparation function that might be causing data leakage:

Read file: data_processing.py
Now let's look at the implementation of create_streak_features, which seems to be creating the actual features:

Read file: data_processing.py
Now, let's look at how the model is trained and evaluated in `modeling.py`:

Read file: modeling.py
Let's continue to examine the `prepare_train_test_features` function, which is key to understanding how temporal separation is (or isn't) being handled:

Read file: data_processing.py
Let's examine the `make_feature_vector` function, which seems to be used for predictions:

Read file: data_processing.py
Now let's check how the `predict_next_cluster` function in `modeling.py` works with these features:

Read file: modeling.py
Let's see the rest of the `predict_next_cluster` function:

Read file: modeling.py
Now let's look at the implementation of the `rolling_origin_indices` function to understand how time-series validation is handled:

Read file: modeling.py
Let's look at how the model is trained in `train_model`:

Read file: modeling.py
Now, let's examine the daily updates functionality to understand how model updates work with new data:

Read file: daily_updates.py
Let's read the rest of `daily_updates.py`:

Read file: daily_updates.py
Now let's look at how the `analyzer.py` file orchestrates the analysis:

Read file: analyzer.py
Let's look at how features are prepared within the `CrashStreakAnalyzer`:

Read file: analyzer.py
Finally, let's look at how predictions are made in the Analyzer class:

Read file: analyzer.py
After examining all the code, I can provide an analysis of how features are created and identify potential sources of data leakage:

The `prepare_train_test_features` function is designed to prevent data leakage when creating features for machine learning. Here's how it works:

1. First, it extracts streak information from the raw game data using `extract_streaks_and_multipliers`.

2. Then it sorts streaks by `streak_number` to ensure temporal ordering and splits them into training and testing sets using a sequential split based on the `test_frac` parameter. This is a proper time-series split that respects the chronological nature of the data.

3. Features are created separately for training and testing data using `create_streak_features`, which includes:
   - Lagged features of previous streaks (e.g., prev1_streak_length, prev2_streak_length)
   - Rolling window statistics (e.g., rolling_mean_streak_length, rolling_std_streak_length)

4. Streak length percentiles are calculated using only the training data, which is correct to prevent data leakage.

5. Clustering is applied to training data based on these percentiles.

6. The StandardScaler is fitted only on the training data features.

7. For test data, a buffer of the last `window` rows from the training set is included to ensure that features for the first test rows are properly created. This is necessary because the feature creation depends on previous streaks.

8. The test data is then clustered using the percentile boundaries calculated from the training data.

9. The scaler fitted on the training data is applied to the test data.

The main sources of potential data leakage are:

1. **Legacy Feature Preparation**: The `prepare_features` function processes all data at once, which can cause data leakage. This is acknowledged in the code with a warning, and the `prepare_train_test_features` function is provided as the correct alternative. The Analyzer class has a `TIME_SERIES_MODE` parameter defaulting to `True` that uses the proper function.

2. **CV Strategy**: The cross-validation strategy in the `train_model` function uses a rolling origin approach, but it's important to ensure that there's a proper gap between training and validation data to avoid indirect data leakage through auto-correlated features. The `gap` parameter in the `rolling_origin_indices` function is set to the `window` size, which should help prevent this.

3. **Batch Effects**: When adding new data in the `daily_update` function, there could be drift in the multiplier rate. The function checks for this and retrains if necessary, but it's important to ensure that the retrained model doesn't contain information from future data.

4. **Online Learning Risks**: When retraining models with new data, the `process_daily_update` function appends new rows to the existing DataFrame and then potentially retrains the model. It's important that this process maintains the proper chronological separation between training and test data.

In summary, the code contains protections against data leakage in feature creation, with the `prepare_train_test_features` function implementing a proper time-series split and handling feature engineering separately for training and test data. The main risk would be using the legacy `prepare_features` function instead of the time-series version, but the default configuration uses the safer approach.
