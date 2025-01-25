import numpy as np
import pandas as pd
from nixtla import NixtlaClient
import os

# TimeGPT key identification
nixtla_client = NixtlaClient(
    api_key="nixak-GTNWCyIqUdHIpGMwLban6vwkZMrhdqGdN6QwM4jo2RddMi4TyjhBxnEab9A8tXO2TU6X0r8jqcbn0IFj"  # Replace with your TimeGPT API key
)

nixtla_client.validate_api_key()

def GWN(df, scale, target_col):
    l = len(df[target_col])
    noise = df.copy()
    noise[target_col] = noise[target_col] + np.random.normal(0, scale, l)
    return noise

def DGA(df, nixtla_client, scale, time_col, target_col, h, freq, mean, std):
    l = len(df[target_col])
    u = (np.random.rand(l) - 0.5) * scale
    
    df_1 = df.copy()
    df_2 = df.copy()
    df_1[target_col] = (df_1[target_col] + u - mean)/std
    df_2[target_col] = (df_2[target_col] - mean)/std
 
    target = np.random.normal(0,1,h)
                              
    timegpt_fcst_df_1 = nixtla_client.forecast(df=df_1, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_1 = timegpt_fcst_df_1['TimeGPT']
    timegpt_fcst_df_2 = nixtla_client.forecast(df=df_2, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_2 = timegpt_fcst_df_2['TimeGPT']
    
    dis_1 = pd.Series(pred_1.values - target)
    dis_2 = pd.Series(pred_2.values - target)
    
    gradient = (dis_1.abs().sum()-dis_2.abs().sum()) / u
    
    noise = df.copy()
    noise[target_col] = noise[target_col] + scale * np.sign(gradient)
    
    return noise

# Setting
time_col = 'ds'
target_col = 'y'
h = 48  # horizon length
n = 96  # input length
model = 'TimeGPT_'
freq = 'h'  # hourly frequency for ETTh1

# Data read
input_file = 'dataset/ETTh2.csv'
dataset_name = os.path.splitext(os.path.basename(input_file))[0]
df = pd.read_csv(input_file)
print(f"Total dataset length: {len(df)}")

# Rename columns from ETTh1 format to our format
df = df.rename(columns={'date': 'ds', 'OT': 'y'})

# Calculate statistics for standardization
std = df[target_col].std()
mean = df[target_col].mean()

# dataset split (0.6, 0.2, 0.2)
l = len(df)
l_train = int(0.6*l)      # 60% for training
l_validation = int(0.2*l)  # 20% for validation
l_test = l - l_train - l_validation  # 20% for testing

train = df.iloc[0:l_train,:]
test = df.iloc[-l_test:,:]  # Use last 20% for testing

# Convert date columns to datetime
train['ds'] = pd.to_datetime(train['ds'])
test['ds'] = pd.to_datetime(test['ds'])

# Scale Setting
scale = mean * 0.02
print('scale:', scale)

iter = int((l_test - h)/h) - 1
print('Number of iterations:', iter)
pred_ = []
truth_ = []
pred_dga = []

for i in range(iter):
    print('iteration number:', i)
    input = test.iloc[i * h : i * h + n,:]
    output = test.iloc[i * h + n: i * h + n + h,:]
    train_ = input.copy()
    train_[target_col] = (train_[target_col] - mean)/std 
    test_ = output.copy()
    test_[target_col] = (test_[target_col] - mean)/std 
    truth_.append(test_[target_col].values.copy())
       
    # Clean prediction
    timegpt_fcst_df = nixtla_client.forecast(df=train_, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizion')
    prediction = timegpt_fcst_df['TimeGPT']
    pred_.append(prediction.values.copy())

    # prediction with dga
    input_dga = DGA(input, nixtla_client, scale, time_col, target_col, h, freq, mean, std)
    input_dga_ = input_dga.copy()
    input_dga_[target_col] = (input_dga_[target_col] - mean)/std 
    timegpt_dga_df = nixtla_client.forecast(df=input_dga_, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizion')

    prediction_dga = timegpt_dga_df['TimeGPT']
    pred_dga.append(prediction_dga.values.copy())

truth_ = np.array(truth_)
pred_ = np.array(pred_)
pred_dga = np.array(pred_dga)

# Calculate and print metrics
mae_pure = np.mean(np.mean(np.abs(pred_ - truth_)))
mse_pure = np.mean((pred_-truth_)**2)
print('prediction error w/o attacks: mae=', mae_pure)
print('prediction error w/o attacks: mse=', mse_pure)

mae_dga = np.mean(np.mean(np.abs(pred_dga - truth_)))
mse_dga = np.mean((pred_dga-truth_)**2)
print('prediction error w/ dga attacks: mae=', mae_dga)
print('prediction error w/ dga attacks: mse=', mse_dga)

# Save results
results = {
    'Model': ['TimeGPT'],
    'Clean_MAE': [mae_pure],
    'Clean_MSE': [mse_pure],
    'Attack_MAE': [mae_dga],
    'Attack_MSE': [mse_dga]
}
results_df = pd.DataFrame(results)
result_filename = f'{dataset_name}_in{n}_out{h}_timegpt_attack_results.csv'
results_df.to_csv(result_filename, index=False)
print(f"\nResults saved to: {result_filename}")
print(results_df)

