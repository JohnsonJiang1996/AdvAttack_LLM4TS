import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient

def GWN(df, scale, target_col):
    l = len(df[target_col])
    noise = df.copy()
    noise[target_col] = noise[target_col] + np.random.normal(0, scale, l)
    return noise

def DGA(df, test, nixtla_client, scale, time_col, target_col, h, freq, mean, std):
    l = len(df[target_col])
    u = (np.random.rand(l) - 0.5) * scale
    
    df_1 = df.copy()
    df_2 = df.copy()
    df_1[target_col] = (df_1[target_col] + u - mean)/std
    df_2[target_col] = (df_2[target_col] - mean)/std
    
    #df_target = pd.DataFrame({
    #time_col: test[time_col],          
    #target_col: np.random.normal(mean,std,h)})
    target = np.random.normal(0,1,h)
                              
    timegpt_fcst_df_1 = nixtla_client.forecast(df=df_1, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_1 = timegpt_fcst_df_1['TimeGPT']
    timegpt_fcst_df_2 = nixtla_client.forecast(df=df_2, h=h, time_col=time_col, target_col=target_col, freq=freq, model='timegpt-1-long-horizon')
    pred_2 = timegpt_fcst_df_2['TimeGPT']
    
    #print(pred_1)
    dis_1 = pd.Series(pred_1.values - target)
    dis_2 = pd.Series(pred_2.values - target)
    
    #dis_1 = pred_1 - df_target[target_col]
    #dis_2 = pred_2 - df_target[target_col]
    #print(dis_1)
    #print(dis_2)
    gradient = (dis_1.abs().sum()-dis_2.abs().sum()) / u
    #print(gradient)
    
    noise = df.copy()
    noise[target_col] = noise[target_col] + scale * np.sign(gradient)
    
    return noise



     
    
def SPSA(df, nixtla_client, scale, time_col, target_col, h):
    l = len(df[target_col])
    u1 = (np.random.rand(l) - 0.5) * scale
    u2 = (np.random.rand(l) - 0.5) * scale
    
    df_1 = df.copy()
    df_1[target_col] = df_1[target_col] + u1
    df_2 = df.copy()
    df_2[target_col] = df_2[target_col] + u2
    
    timegpt_fcst_df_1 = nixtla_client.forecast(df=df_1, h=h, time_col=time_col, target_col=target_col, freq='MS', model='timegpt-1-long-horizon')
    pred_1 = timegpt_fcst_df_1['TimeGPT']
    timegpt_fcst_df_2 = nixtla_client.forecast(df=df_2, h=h, time_col=time_col, target_col=target_col, freq='MS', model='timegpt-1-long-horizon')
    pred_2 = timegpt_fcst_df_2['TimeGPT']
    
    gradient = (pred_1 - pred_2).abs().sum() / (u1-u2)
    #print(gradient)
    
    noise = df.copy()
    noise[target_col] = noise[target_col] + scale * np.sign(gradient)
    
    return noise

def ite(df, nixtla_client, scale, time_col, target_col, h, epoch, alpha=1):
    l = len(df[target_col])
    u1 = (np.random.rand(l) - 0.5) 
    u2 = (np.random.rand(l) - 0.5) 
    
    df_1 = df.copy()
    df_2 = df.copy()
            
    for i in range(epoch):
        
        df_1[target_col] = df[target_col] + u1
        df_2[target_col] = df[target_col] + u2
    
        timegpt_fcst_df_1 = nixtla_client.forecast(df=df_1, h=h, time_col=time_col, target_col=target_col, freq='MS', model='timegpt-1-long-horizon')
        pred_1 = timegpt_fcst_df_1['TimeGPT']
        timegpt_fcst_df_2 = nixtla_client.forecast(df=df_2, h=h, time_col=time_col, target_col=target_col, freq='MS', model='timegpt-1-long-horizon')
        pred_2 = timegpt_fcst_df_2['TimeGPT']
    
        gradient = (pred_1 - pred_2).abs().sum() / (u1-u2)
        u1 = u1 + alpha * np.sign(gradient)
        #u1[np.where(u1>scale)] = scale
        #u1[np.where(u1<-1*scale)] = -1*scale
        u2 = u2 - alpha * np.sign(gradient)
        #u2[np.where(u2>scale)] = scale
        #u2[np.where(u2<-1*scale)] = -1*scale
        
    
    noise = df.copy()
    u1[np.where(u1>scale)] = scale
    u1[np.where(u1<-1*scale)] = -1*scale
    noise[target_col] = noise[target_col] + u1
    #noise[target_col] = noise[target_col] + scale * np.sign(gradient)
    
    return noise