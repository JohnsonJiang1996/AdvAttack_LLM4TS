import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from neuralforecast import NeuralForecast
from sklearn.preprocessing import StandardScaler
from neuralforecast.models import LSTM, NHITS, Autoformer, iTransformer, PatchTST, TimesNet, NLinear, Informer, TimeLLM
from nixtla import NixtlaClient
import numpy as np
import os

# Enable Tensor Cores optimization
torch.set_float32_matmul_precision('high')

def GWN(df, scale, target_col):
    l = len(df[target_col])
    noise = df.copy()
    noise[target_col] = noise[target_col] + np.random.normal(0, scale, l)
    return noise

def DGA(df, nixtla_client, scale, time_col, target_col, h, freq, mean, std):
    l = len(df[target_col])
    u = (np.random.rand(l) - 0.5) * scale
    
    # For gradient calculation, we need standardized data
    df_1 = df.copy()
    df_2 = df.copy()
    df_1[target_col] = (df_1[target_col] - mean)/std
    df_2[target_col] = (df_2[target_col] - mean)/std
    
    # Add perturbation to standardized data for gradient calculation
    df_1[target_col] = df_1[target_col] + u/std  # Divide by std because we're in standardized space
 
    target = np.random.normal(0,1,h)
                              
    timegpt_fcst_df_1 = nixtla_client.forecast(df=df_1, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_1 = timegpt_fcst_df_1['TimeGPT']
    timegpt_fcst_df_2 = nixtla_client.forecast(df=df_2, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_2 = timegpt_fcst_df_2['TimeGPT']
    
    dis_1 = pd.Series(pred_1.values - target)
    dis_2 = pd.Series(pred_2.values - target)
    
    gradient = (dis_1.abs().sum()-dis_2.abs().sum()) / (u/std)  # Use u/std to match the perturbation scale
    
    # Add perturbation to original (non-standardized) data
    noise = df.copy()
    noise[target_col] = noise[target_col] + scale * np.sign(gradient)
    
    return noise

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Initialize TimeGPT client
nixtla_client = NixtlaClient(
    api_key="nixak-GTNWCyIqUdHIpGMwLban6vwkZMrhdqGdN6QwM4jo2RddMi4TyjhBxnEab9A8tXO2TU6X0r8jqcbn0IFj"  # Replace with your TimeGPT API key
)
nixtla_client.validate_api_key()

# Load the ETTh1 dataset
input_file = 'dataset/ETTh1.csv'
dataset_name = os.path.splitext(os.path.basename(input_file))[0]  # Extract dataset name from file path
df = pd.read_csv(input_file)
print(f"Total dataset length: {len(df)}")

# Rename columns
df = df.rename(columns={'date': 'ds', 'OT': 'y'})

# Add unique_id column
df['unique_id'] = 1

# Split data into train and test sets (0.6 for train, 0.2 for test from the end)
total_length = len(df)
train_size = int(total_length * 0.6)
test_size = int(total_length * 0.2)

# Get training and test data
Y_train = df.iloc[:train_size].copy()
test_df = df.iloc[-test_size:].copy()

print(f"Train set length: {len(Y_train)}")
print(f"Test set length: {len(test_df)}")

# Convert date columns to datetime type
Y_train['ds'] = pd.to_datetime(Y_train['ds'])
test_df['ds'] = pd.to_datetime(test_df['ds'])

# Create a StandardScaler instance
scaler = StandardScaler()

# Get mean/std for standardization and attack
mean = Y_train['y'].mean()
std = Y_train['y'].std()
scale = mean * 0.05  # Same scale as in main.py

# Only standardize training data for other models
Y_train['y'] = scaler.fit_transform(Y_train['y'].values.reshape(-1, 1))

# Set model hyperparameters
horizon = 48  # Forecast length
input_length = 96  # Input sequence length
train_steps = 10  # Training steps
freq = 'h'  # hourly frequency for ETTh1

# Define the list of models
models = [
    LSTM(h=horizon, max_steps=train_steps, scaler_type='standard', encoder_hidden_size=64, decoder_hidden_size=64),
    NHITS(h=horizon, input_size=input_length, max_steps=train_steps, n_freq_downsample=[2, 1, 1]),
    iTransformer(h=horizon, n_series=1, input_size=input_length, max_steps=train_steps),
    PatchTST(h=horizon, input_size=input_length, max_steps=train_steps),
    TimesNet(h=horizon, input_size=input_length, max_steps=train_steps),
    NLinear(h=horizon, input_size=input_length, max_steps=train_steps),
    'TimeGPT'
]

# Define tables to save the results for both clean and attacked data
results = {'Model': [], 'Clean_MAE': [], 'Clean_MSE': [], 'Attack_MAE': [], 'Attack_MSE': []}

# Perform rolling predictions, evaluating each model one by one
for model in models:
    model_name = model if isinstance(model, str) else type(model).__name__
    print(f"Evaluating model: {model_name}")
    
    clean_mae_list = []
    clean_mse_list = []
    attack_mae_list = []
    attack_mse_list = []

    if model_name == 'TimeGPT':
        # Calculate iterations like main.py
        iter = int((len(test_df) - horizon)/horizon) - 1
        print(f"Number of iterations for {model_name}: {iter}")
        
        for i in range(iter):
            # Extract input and output windows
            input_df = test_df.iloc[i * horizon : i * horizon + input_length].copy()
            output_df = test_df.iloc[i * horizon + input_length : i * horizon + input_length + horizon].copy()
            
            if len(input_df) < input_length or len(output_df) < horizon:
                print(f"Warning: Window size too small at iteration {i}, skipping")
                continue
            
            # Clean prediction
            input_clean = input_df.copy()
            input_clean['y'] = (input_clean['y'] - mean)/std
            timegpt_preds_df = nixtla_client.forecast(df=input_clean, h=horizon, freq=freq, time_col='ds', target_col='y', model='timegpt-1-long-horizon')
            clean_pred = timegpt_preds_df['TimeGPT'].values
            
            # Attack prediction
            input_dga = DGA(input_df, nixtla_client, scale, 'ds', 'y', horizon, freq, mean, std)
            # Standardize after DGA
            input_dga['y'] = (input_dga['y'] - mean)/std
            timegpt_dga_df = nixtla_client.forecast(df=input_dga, h=horizon, freq=freq, time_col='ds', target_col='y', model='timegpt-1-long-horizon')
            attack_pred = timegpt_dga_df['TimeGPT'].values

            # Standardize true values
            true_values = (output_df['y'] - mean)/std

            # Skip if NaN values present
            if np.isnan(clean_pred).any() or np.isnan(attack_pred).any() or np.isnan(true_values).any():
                continue

            # Calculate errors
            clean_mae_list.append(mean_absolute_error(true_values, clean_pred))
            clean_mse_list.append(mean_squared_error(true_values, clean_pred))
            attack_mae_list.append(mean_absolute_error(true_values, attack_pred))
            attack_mse_list.append(mean_squared_error(true_values, attack_pred))

    else:
        # Train the model on standardized training data
        nf = NeuralForecast(models=[model], freq='h')
        nf.fit(df=Y_train)

        # Original iteration logic for other models
        for i in range(0, len(test_df) - input_length - horizon, horizon):
            # Extract input data
            input_df = test_df.iloc[i:i + input_length].copy()
            output_df = test_df.iloc[i + input_length:i + input_length + horizon].copy()
            
            if len(input_df) < input_length or len(output_df) < horizon:
                print(f"Warning: Window size too small at iteration {i}, skipping")
                continue

            # Clean prediction
            input_clean = input_df.copy()
            input_clean['y'] = scaler.transform(input_clean['y'].values.reshape(-1, 1))
            clean_preds = nf.predict(df=input_clean)
            clean_pred = clean_preds[model_name].values
            
            # Attack prediction using GWN
            input_gwn = GWN(input_df, scale, 'y')  # Use GWN instead of simple_DGA
            input_gwn['y'] = scaler.transform(input_gwn['y'].values.reshape(-1, 1))  # Standardize after adding noise
            attack_preds = nf.predict(df=input_gwn)
            attack_pred = attack_preds[model_name].values

            # Extract and standardize true values
            true_values = (output_df['y'] - mean)/std

            # Skip if NaN values present
            if np.isnan(clean_pred).any() or np.isnan(attack_pred).any() or np.isnan(true_values).any():
                continue

            # Calculate errors
            clean_mae_list.append(mean_absolute_error(true_values, clean_pred))
            clean_mse_list.append(mean_squared_error(true_values, clean_pred))
            attack_mae_list.append(mean_absolute_error(true_values, attack_pred))
            attack_mse_list.append(mean_squared_error(true_values, attack_pred))

    # Store results
    results['Model'].append(model_name)
    results['Clean_MAE'].append(np.mean(clean_mae_list) if clean_mae_list else np.nan)
    results['Clean_MSE'].append(np.mean(clean_mse_list) if clean_mse_list else np.nan)
    results['Attack_MAE'].append(np.mean(attack_mae_list) if attack_mae_list else np.nan)
    results['Attack_MSE'].append(np.mean(attack_mse_list) if attack_mse_list else np.nan)

# Save results to a DataFrame and display
results_df = pd.DataFrame(results)
# Create dynamic filename with dataset name, input length and horizon
result_filename = f'{dataset_name}_in{input_length}_out{horizon}_with_attack_results.csv'
results_df.to_csv(result_filename, index=False)
print(f"Results saved to: {result_filename}")
print("\nResults:")
print(results_df)
