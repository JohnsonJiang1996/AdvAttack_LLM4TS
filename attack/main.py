import numpy as np
import pandas as pd
from nixtla import NixtlaClient

# TimeGPT key identification
nixtla_client = NixtlaClient(
    api_key = # add the key from TimeGPT. https://docs.nixtla.io/docs/getting-started-setting_up_your_api_key
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
time_col = 'date'
target_col = 'Australia'
h = 48
n = 96
model = 'TimeGPT_'

# Data read
ds_name = 'exchange2' + '.csv'
df = pd.read_csv(ds_name)
print(df.columns)
std = df[target_col].std()
mean = df[target_col].mean()


# dataset split
l = len(df)
l_train = int(0.5*l)
l_validation = int(0.25*l)
l_test = l - l_train - l_validation
freq = 'D'
train = df.iloc[0:l_train,:]
test = df.iloc[l_train + l_validation : l,:]


# Scale Setting
scale = mean * 0.02
print('scale:', scale)

iter = int((l_test - h)/h) - 1
print(iter)
pred_ = []
truth_ = []
#pred_gwn = []
pred_dga = []

for i in range(iter):
    print('iteration number:',i)
    input = test.iloc[i * h : i * h + n,:]
    output = test.iloc[i * h + n: i * h + n + h,:]
    train_ = input.copy()
    train_[target_col] = (train_[target_col] - mean)/std 
    test_ = output.copy()
    test_[target_col] = (test_[target_col] - mean)/std 
    truth_.append(test_[target_col].values.copy())
       
    timegpt_fcst_df = nixtla_client.forecast(df=train_, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    timegpt_fcst_df.head()

    prediction = timegpt_fcst_df['TimeGPT']
    pred_.append(prediction.values.copy())

    # prediction with GWN
    #input_gwn = GWN(input, scale, target_col)
    #input_gwn_ = input_gwn.copy()
    #input_gwn_[target_col] = (input_gwn_[target_col] - mean)/std 
    #timegpt_gwn_df = nixtla_client.forecast(df=input_gwn_, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    #timegpt_gwn_df.head()
    #prediction_gwn = timegpt_gwn_df['TimeGPT']
    #pred_gwn.append(prediction_gwn.values.copy())

    # prediction with dga
    input_dga = DGA(input, nixtla_client, scale, time_col, target_col, h, freq, mean, std)
    input_dga_ = input_dga.copy()
    input_dga_[target_col] = (input_dga_[target_col] - mean)/std 
    timegpt_dga_df = nixtla_client.forecast(df=input_dga_, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    timegpt_dga_df.head()

    prediction_dga = timegpt_dga_df['TimeGPT']
    pred_dga.append(prediction_dga.values.copy())

truth_ = np.array(truth_)
pred_ = np.array(pred_)
#pred_gwn = np.array(pred_gwn)
pred_dga = np.array(pred_dga)

mae_pure = np.mean(np.mean(np.abs(pred_ - truth_)))
mse_pure = np.mean((pred_-truth_)**2)
print('prediction error w/o attacks: mae=',mae_pure)
print('prediction error w/o attacks: mse=',mse_pure)

#mae_gwn = np.mean(np.mean(np.abs(pred_gwn - truth_)))
#mse_gwn = np.mean((pred_gwn-truth_)**2)
#print('prediction error w/ gwns: mae=',mae_gwn)
#print('prediction error w/ gwns: mse=',mse_gwn)

mae_dga = np.mean(np.mean(np.abs(pred_dga - truth_)))
mse_dga = np.mean((pred_dga-truth_)**2)
print('prediction error w/ dga attacks: mae=',mae_dga)
print('prediction error w/ dga attacks: mse=',mse_dga)

