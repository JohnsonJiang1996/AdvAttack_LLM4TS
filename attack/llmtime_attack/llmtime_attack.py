import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.validation_likelihood_tuning import get_autotuned_predictions_data

def GWN(A, scale):
    noise = A.copy()
    noise = noise + np.random.normal(0, scale, len(noise)) 
    return noise

def SPSA(train, test, hypers, num_samples, model, scale, mean, std, z_score_flag):
    l = len(train)
    u1 = (np.random.rand(l) - 0.5) * scale
    if z_score_flag:
        train_1 = (train + u1 - mean)/std
        train_2 = (train - u1 - mean)/std
    else:
        train_1 = train + u1
        train_2 = train - u1
    pred_dict1 = get_autotuned_predictions_data(train_1, test, hypers, num_samples, model, verbose=False, parallel=False)
    pred1 = pred_dict1['median']
    pred1 = pd.Series(pred1, index=test.index)
    pred_dict2 = get_autotuned_predictions_data(train_2, test, hypers, num_samples, model, verbose=False, parallel=False)
    pred2 = pred_dict2['median']
    pred2 = pd.Series(pred2, index=test.index)
    gradient = (pred1 - pred2).abs().sum() / (2*u1)
    noise = train.copy()
    noise = noise - scale * np.sign(gradient)
    return noise

def BIM(train, test, hypers, num_samples, model, scale, mean, std, alpha, N=5, z_score_flag=False):
    l = len(train)
    train_update = train
    for i in range(N):
        u1 = (np.random.rand(l) - 0.5) * scale
        if z_score_flag:
            train_1 = (train_update + u1 - mean)/std
            train_2 = (train_update - u1 - mean)/std
        else:
            train_1 = train_update + u1
            train_2 = train_update - u1
    
        pred_dict1 = get_autotuned_predictions_data(train_1, test, hypers, num_samples, model, verbose=False, parallel=False)
        pred1 = pred_dict1['median']
        pred1 = pd.Series(pred1, index=test.index)
        pred_dict2 = get_autotuned_predictions_data(train_2, test, hypers, num_samples, model, verbose=False, parallel=False)
        pred2 = pred_dict2['median']
        pred2 = pd.Series(pred2, index=test.index)
        gradient = (pred1 - pred2).abs().sum() / (2*u1)
        train_update = train_update + alpha * np.sign(gradient)
        train_update.clip(upper = train + scale)
        train_update.clip(lower = train - scale)
    return train_update


def memory_reshape(train, test, hypers, num_samples, model):
    l = len(train)
    l_test = len(test)
    freq = train.index.freq
    new_dates = pd.date_range(train.index[-1] + freq, periods = l, freq = freq)
    new_values = pd.Series(train.values, index = new_dates)
    train_ = pd.concat([train, new_values])
    test_ = train_[l+1:l+l_test]
    #pred_dict = get_autotuned_predictions_data(train, test, hypers, num_samples, model, verbose=False, parallel=False)
    
    pred_dict = get_autotuned_predictions_data(train_, test_, hypers, num_samples, model, verbose=False, parallel=False)
    return pred_dict

def DGA(train, test, hypers, num_samples, model, scale, mean, std, alpha, N=5, z_score_flag=False):
    l = len(train)
    l_test = len(test)
    u1 = (np.random.rand(l) - 0.5) * alpha
    if z_score_flag:
        target = np.random.normal(0,1,l_test)
    else:
        target = np.random.normal(mean, std, l_test)
    train_update = train + u1
    train_update_2 = train
            
    for i in range(N):
        if z_score_flag:
            train_1 = (train_update - mean)/std
            train_2 = (train_update_2 - mean)/std
        else:
            train_1 = train_update
            train_2 = train_update_2
    
        pred_dict2 = get_autotuned_predictions_data(train_2, test, hypers, num_samples, model, verbose=False, parallel=False)
        pred2 = pred_dict2['median']
        pred2 = pd.Series(pred2, index=test.index)
        diff2 = (target - pred2).abs().sum()         
    
        pred_dict1 = get_autotuned_predictions_data(train_1, test, hypers, num_samples, model, verbose=False, parallel=False)
        pred1 = pred_dict1['median']
        pred1 = pd.Series(pred1, index=test.index)
        gradient =( (target - pred1).abs().sum() - diff2)/ (u1)
        
        train_update_2 = train_update
        
        train_update = train_update + alpha * np.sign(gradient)
        train_update.clip(upper = train + scale)
        train_update.clip(lower = train - scale)
        
    return train_update