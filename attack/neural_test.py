import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from neuralforecast import NeuralForecast
from sklearn.preprocessing import StandardScaler
from neuralforecast.models import LSTM, NHITS, Autoformer, iTransformer, PatchTST, TimesNet, NLinear, Informer, TimeLLM
from nixtla import NixtlaClient

import numpy as np
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize TimeGPT client
nixtla_client = NixtlaClient(
    api_key="YOUR_API_KEY"  # Replace with your TimeGPT API key
)
nixtla_client.validate_api_key()

# Load the standardized ETTh1 dataset
target_column = 'y'

# Load training and test sets
Y_train = pd.read_csv('exchange2_train.csv')
test_df = pd.read_csv('exchange2_test.csv')

# Convert date columns to datetime type
Y_train['ds'] = pd.to_datetime(Y_train['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])
Y_train['unique_id'] = 1
test_df['unique_id'] = 1

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit standardization on the training set
scaler.fit(Y_train[target_column].values.reshape(-1, 1))

# Apply standardization to both the training and test sets
Y_train[target_column] = scaler.transform(Y_train[target_column].values.reshape(-1, 1))
test_df[target_column] = scaler.transform(test_df[target_column].values.reshape(-1, 1))

# Only use the first 14400 rows
# Y_df = Y_df.iloc[:14400].copy()

# Define training, validation, and test set splits
# train_size = int(len(Y_df) * 0.50)  # 50% for training
# test_size = int(len(Y_df) * 0.25)   # Last 25% for testing
# train_df = Y_df.iloc[:train_size].copy()  # First 50% for training
# test_df = Y_df.iloc[-test_size:].copy()   # Last 25% for testing

# Set model hyperparameters
horizon = 48  # Forecast length
input_length = 96  # Input sequence length
# Y_train = Y_train.sample(frac=0.5, random_state=42)
# test_df = test_df.sample(frac=0.5, random_state=42)
train_steps = 10  # Training steps

# Define the list of models
models = [
    LSTM(h=horizon, max_steps=train_steps, scaler_type='standard', encoder_hidden_size=64, decoder_hidden_size=64),
    NHITS(h=horizon, input_size=input_length, max_steps=train_steps, n_freq_downsample=[2, 1, 1]),
    Autoformer(h=horizon, input_size=input_length, max_steps=train_steps),
    iTransformer(h=horizon, n_series=1, input_size=input_length, max_steps=train_steps),
    PatchTST(h=horizon, input_size=input_length, max_steps=train_steps),
    TimesNet(h=horizon, input_size=input_length, max_steps=train_steps),
    NLinear(h=horizon, input_size=input_length, max_steps=train_steps),
    Informer(h=horizon, input_size=input_length, max_steps=train_steps), 'TimeGPT'
]
# TimeLLM(h=horizon, input_size=input_length,batch_size=8,
#       valid_batch_size=8,
#       windows_batch_size=256,
#       inference_windows_batch_size=256,),]

# Add a Historical Average method

# Define a table to save the results
results = {'Model': [], 'MAE': [], 'MSE': []}

# Perform rolling predictions, evaluating each model one by one
for model in models:
    model_name = model if isinstance(model, str) else type(model).__name__
    print(f"Evaluating model: {model_name}")

    if model_name == 'HistoricalAverage':
        mae_list = []
        mse_list = []

        # Rolling window prediction
        for i in range(0, len(test_df) - input_length - horizon, input_length):
            # Extract input data with length of input_length
            input_df = test_df.iloc[i:i + input_length].copy()

            # Compute the historical average of the input
            hist_avg = input_df['y'].mean()

            # Generate predictions of length horizon, with values all equal to the historical average
            pred_values = np.full(horizon, hist_avg)

            # Extract the true values for the corresponding time period
            true_values = test_df['y'].iloc[i + input_length:i + input_length + horizon].values  # Extract corresponding true values

            # Check if the true values contain NaN
            if np.isnan(true_values).any():
                print(f"NaN detected in true values at step {i}")
                continue  # Skip steps that contain NaN

            # Calculate the error
            mae = mean_absolute_error(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)

            # Record the error
            mae_list.append(mae)
            mse_list.append(mse)

        # Compute the average MAE and MSE
        avg_mae = np.mean(mae_list) if mae_list else np.nan
        avg_mse = np.mean(mse_list) if mse_list else np.nan

        # Store the results
        results['Model'].append(model_name)
        results['MAE'].append(avg_mae)
        results['MSE'].append(avg_mse)

    elif model_name == 'TimeGPT':
        mae_list = []
        mse_list = []

        # Rolling window prediction
        for i in range(0, len(test_df) - input_length - horizon, input_length):
            # Extract input data with length of input_length
            input_df = test_df.iloc[i:i + input_length].copy()

            # Use TimeGPT for prediction
            timegpt_preds_df = nixtla_client.forecast(df=input_df, h=horizon, freq='D', time_col='ds', target_col='y', model='timegpt-1-long-horizon')

            # Extract TimeGPT predictions
            pred_values = timegpt_preds_df['TimeGPT'].values

            # Extract the true values for the corresponding time period
            true_values = test_df['y'].iloc[i + input_length:i + input_length + horizon].values

            # Check if the predicted or true values contain NaN
            if np.isnan(pred_values).any() or np.isnan(true_values).any():
                print(f"NaN detected in predicted or true values for TimeGPT at step {i}")
                continue  # Skip steps that contain NaN

            # Calculate the error
            mae = mean_absolute_error(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)

            # Record the error
            mae_list.append(mae)
            mse_list.append(mse)

        # Compute the average MAE and MSE
        avg_mae = np.mean(mae_list) if mae_list else np.nan
        avg_mse = np.mean(mse_list) if mse_list else np.nan

        # Store the results
        results['Model'].append(model_name)
        results['MAE'].append(avg_mae)
        results['MSE'].append(avg_mse)

    else:
        # Train and predict for other models
        nf = NeuralForecast(models=[model], freq='D')
        nf.fit(df=Y_train)  # Train the model on the training data

        mae_list = []
        mse_list = []

        # Rolling window prediction
        for i in range(0, len(test_df) - input_length - horizon, horizon):
            # Extract input data with length of input_length
            input_df = test_df.iloc[i:i + input_length].copy()

            # Use the trained model for prediction
            preds_df = nf.predict(df=input_df)  # Predict using the model

            # Extract the model predictions
            pred_values = preds_df[model_name].values

            # Check if the predicted values contain NaN
            if np.isnan(pred_values).any():
                print(f"NaN detected in predicted values for {model_name} at step {i}")
                continue  # Skip steps that contain NaN

            # Extract the true values for the corresponding time period
            true_values = test_df['y'].iloc[i + input_length:i + input_length + horizon].values  # Extract corresponding true values

            # Check if the true values contain NaN
            if np.isnan(true_values).any():
                print(f"NaN detected in true values at step {i}")
                continue  # Skip steps that contain NaN

            # Calculate the error
            mae = mean_absolute_error(true_values, pred_values)
            mse = mean_squared_error(true_values, pred_values)

            # Record the error
            mae_list.append(mae)
            mse_list.append(mse)

        # Compute the average MAE and MSE
        avg_mae = np.mean(mae_list) if mae_list else np.nan
        avg_mse = np.mean(mse_list) if mse_list else np.nan

        # Store the results
        results['Model'].append(model_name)
        results['MAE'].append(avg_mae)
        results['MSE'].append(avg_mse)

# Save results to a DataFrame and display
results_df = pd.DataFrame(results)
results_df.to_csv('exchange_results_with_timegpt_48_new.csv', index=False)
print(results_df)
