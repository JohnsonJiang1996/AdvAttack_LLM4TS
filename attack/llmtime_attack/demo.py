import os
import torch
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
from mistralai import Mistral
api_key = os.environ["MISTRAL_KEY"]
from data.serialize import SerializerSettings
from models.utils import grid_iter
from models.promptcast import get_promptcast_predictions_data
from models.darts import get_arima_predictions_data
from models.llmtime import get_llmtime_predictions_data
from data.small_context import get_datasets
from models.validation_likelihood_tuning import get_autotuned_predictions_data
from attack import GWN
from attack import SPSA 
from attack import DGA
from attack import BIM

def plot_preds(train, test, pred_dict, model_name, show_samples=False):
    pred = pred_dict['median']
    pred = pd.Series(pred, index=test.index)
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(train)
    plt.plot(test, label='Truth', color='black')
    plt.plot(pred, label=model_name, color='purple')
    # shade 90% confidence interval
    samples = pred_dict['samples']
    lower = np.quantile(samples, 0.05, axis=0)
    upper = np.quantile(samples, 0.95, axis=0)
    plt.fill_between(pred.index, lower, upper, alpha=0.3, color='purple')
    if show_samples:
        samples = pred_dict['samples']
        # convert df to numpy array
        samples = samples.values if isinstance(samples, pd.DataFrame) else samples
        for i in range(min(10, samples.shape[0])):
            plt.plot(pred.index, samples[i], color='purple', alpha=0.3, linewidth=1)
    plt.legend(loc='upper left')
    if 'NLL/D' in pred_dict:
        nll = pred_dict['NLL/D']
        if nll is not None:
            plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()
    plt.savefig('output_graphics/testplot4.png')



print(torch.cuda.max_memory_allocated())
print()

gpt4_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

mistral_api_hypers = dict(
    alpha=0.3,
    basic=True,
    temp=1.0,
    top_p=0.8,
    settings=SerializerSettings(base=10, prec=3, signed=True, time_sep=', ', bit_sep='', minus_sign='-')
)

gpt3_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)


llma2_hypers = dict(
    temp=0.7,
    alpha=0.95,
    beta=0.3,
    basic=False,
    settings=SerializerSettings(base=10, prec=3, signed=True, half_bin_correction=True)
)


promptcast_hypers = dict(
    temp=0.7,
    settings=SerializerSettings(base=10, prec=0, signed=True, 
                                time_sep=', ',
                                bit_sep='',
                                plus_sign='',
                                minus_sign='-',
                                half_bin_correction=False,
                                decimal_point='')
)

arima_hypers = dict(p=[12,30], d=[1,2], q=[0])

model_hypers = {
     'LLMTime GPT-3.5': {'model': 'gpt-3.5-turbo-instruct', **gpt3_hypers},
     'LLMTime GPT-4': {'model': 'gpt-4', **gpt4_hypers},
     'LLMTime GPT-3': {'model': 'text-davinci-003', **gpt3_hypers},
     'LLMTime GPT-4o': {'model': 'gpt-4o-2024-08-06', **gpt4_hypers},
     'LLMTime GPT-4o-mini': {'model': 'gpt-4o-mini', **gpt4_hypers},
     'PromptCast GPT-3': {'model': 'text-davinci-003', **promptcast_hypers},
     'LLMA2': {'model': 'llama-7b', **llma2_hypers},
     'mistral': {'model': 'mistral', **llma2_hypers},
     'mistral-api-tiny': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'mistral-api-small': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'mistral-api-medium': {'model': 'mistral-api-tiny', **mistral_api_hypers},
     'ARIMA': arima_hypers,
    
 }


model_predict_fns = {
    #'LLMA2': get_llmtime_predictions_data,
    #'mistral': get_llmtime_predictions_data,
    #'LLMTime GPT-4': get_llmtime_predictions_data,
    #'LLMTime GPT-4o': get_llmtime_predictions_data,
    #'LLMTime GPT-4o-mini': get_llmtime_predictions_data,
    #'mistral-api-tiny': get_llmtime_predictions_data
    'LLMTime GPT-3.5': get_llmtime_predictions_data
}


model_names = list(model_predict_fns.keys())

# data reading
#datasets = get_datasets()
#ds_name = 'AirPassengersDataset'
#data = datasets[ds_name]
#train, test = data # or change to your own data
#

ds_root = 'dataset/'
#ds_name = 'ETTh1'
#ds_name = 'ETTh2'
ds_name = 'weather'
#ds_name = 'IstanbulTraffic'
#ds_name = 'exchange'
ds_location = ds_root + ds_name + '.csv'


if ds_name == 'exchange':
    df = pd.read_csv(ds_location, header=None)
    data = pd.Series(df.iloc[:,0].values, index=df.index)
    std = df.iloc[:,0].std()
    mean = df.iloc[:,0].mean()
else:
    df = pd.read_csv(ds_location)
    data = pd.Series(df['OT'].values, index=df['date'])
    std = df['OT'].std()
    mean = df['OT'].mean()

# Statistics
z_score_flag = True
print(std)
print(mean)

# Setting
scale = mean * 0.02
print('scale:', scale)

if ds_name == 'IstanbulTraffic':
    l_train = 48
    l_validation = 0
    l_test = 144
elif ds_name == 'exchange':
    l_train = 4*960
    l_validation = 2*960
    l_test = 2*960
else:
    l_train = 12 * 30 * 24
    l_validation = 4 * 30 * 24
    l_test = 5 * 30 * 24


train = data[0:l_train]
test = data[l_train + l_validation : l_train + l_validation + l_test]
out = {}

pure_flag = 1
gwn_flag = 1
spsa_flag = 1
bim_flag = 0
dga_flag = 0


#test 
for model in model_names: # GPT-4 takes a about a minute to run
    model_hypers[model].update({'dataset_name': ds_name}) # for promptcast
    hypers = list(grid_iter(model_hypers[model]))
    num_samples = 3
    historical_n = 96
    future_n = 48
    #train_ = (train - mean)/std
    for i in range(20):
        input = test[i * historical_n : (i+1)*historical_n]
        if z_score_flag :
            train_ = (input - mean)/std 
            test_ = (test[(i+1) * historical_n: (i+1) * historical_n + future_n] - mean)/std
        else:
            train_ = input 
            test_ = test[(i+1) * historical_n: (i+1) * historical_n + future_n]
        #print(train_)
        #print(test_) 
        
        # generate prediction
        if pure_flag == 1:
            pred_dict = get_autotuned_predictions_data(train_, test_, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            out[model] = pred_dict
            pred = pred_dict['median']
            print(pred)
            pred = pd.Series(pred, index=test_.index)
            save_name = 'output/' + model + ds_name + '_pure_' + str(i) + '.csv'
            pred.to_csv(save_name, header=True)
        
            #print(str(i) + '_pure forecasting:')
            #print(pred)
            mae_pure = (pred - test_).abs().mean()
            print(str(i) + '_mae of pure forecasting:',mae_pure)
    
        # attack by gwn
        '''
        if gwn_flag == 1:
            gwn = GWN(input, scale)
            if z_score_flag:
                train_gwn = (gwn - mean)/std
            else:
                train_gwn = gwn
            pred_dict_gwn = get_autotuned_predictions_data(train_gwn, test_, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            pred_gwn = pred_dict_gwn['median']
            pred_gwn = pd.Series(pred_gwn, index=test_.index)
            save_name_gwn = 'output/' + model + ds_name + '_gwn_' + str(i) + '.csv'
            pred_gwn.to_csv(save_name_gwn, header=True)
        
        
            #print(str(i) + '_forecasting with gwn:')
            #print(pred_gwn)
            mae_gwn_input = (gwn - input).abs().mean()
            print(str(i) + '_mae of input with gwn:',mae_gwn_input)
            mae_gwn = (pred_gwn - test_).abs().mean()
            print(str(i) + '_mae of forecasting with gwn:',mae_gwn)
        '''
    
        # attack by SPSA
        #'''
        if spsa_flag == 1:
            spsa = SPSA(input, test_, hypers, num_samples, model_predict_fns[model], scale, mean, std, z_score_flag)
            if z_score_flag :
                train_spsa = (spsa - mean)/std
            else:
                train_spsa = spsa
        
            pred_dict_spsa = get_autotuned_predictions_data(train_spsa, test_, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            pred_spsa = pred_dict_spsa['median']
            pred_spsa = pd.Series(pred_spsa, index=test_.index)
            save_name_spsa = 'output/' + model + ds_name + '_spsa_' + str(i) + '.csv'
            pred_spsa.to_csv(save_name_spsa, header=True)
        
        
            #print(str(i) + '_forecasting with spsa:')
            #print(pred_spsa)
            mae_spsa_input = (spsa - input).abs().mean()
            print(str(i) + 'mae of input with spsa:',mae_spsa_input)
            mae_spsa = (pred_spsa - test_).abs().mean()
            print(str(i) + 'mae of forecasting with spsa:',mae_spsa)
        #'''
        
        # attack by dga
        '''
        if dga_flag == 1:
            N = 5
            alpha = scale/2
            dga = DGA(input, test_, hypers, num_samples, model_predict_fns[model], scale, mean, std, alpha, N, z_score_flag)
            if z_score_flag:
                train_dga = (dga - mean)/std
            else:
                train_dga = dga
            pred_dict_dga = get_autotuned_predictions_data(train_dga, test_, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            pred_dga = pred_dict_dga['median']
            pred_dga = pd.Series(pred_dga, index=test_.index)
            save_name_dga = 'output/' + model + ds_name + '_dga_' + str(i) + '.csv'
            pred_dga.to_csv(save_name_dga, header=True)

        
            #print(str(i) + '_forecasting with dga:')
            #print(pred_dga)
            mae_dga_input = (dga - input).abs().mean()
            print(str(i) + 'mae of input with dga:',mae_dga_input)
            mae_dga = (pred_dga - test_).abs().mean()
            print(str(i) + 'mae of forecasting with dga:',mae_dga)

        '''
        
        # attack by bim
        '''
        if bim_flag == 1:
            N = 5
            alpha = scale/2
            bim = BIM(input, test_, hypers, num_samples, model_predict_fns[model], scale, mean, std, alpha, N, z_score_flag)
            if z_score_flag:
                train_bim = (bim - mean)/std
            else:
                train_bim = bim
            pred_dict_bim = get_autotuned_predictions_data(train_bim, test_, hypers, num_samples, model_predict_fns[model], verbose=False, parallel=False)
            pred_bim = pred_dict_bim['median']
            pred_bim = pd.Series(pred_bim, index=test_.index)
            save_name_bim = 'output/' + model + ds_name + '_bim_' + str(i) + '.csv'
            pred_bim.to_csv(save_name_bim, header=True)

        
            #print(str(i) + '_forecasting with bim:')
            #print(pred_bim)
            mae_bim_input = (bim - input).abs().mean()
            print(str(i) + 'mae of input with bim:',mae_bim_input)
            mae_bim = (pred_bim - test_).abs().mean()
            print(str(i) + 'mae of forecasting with bim:',mae_bim)
        '''
    
    # plot the results
    #pred = pred_dict['median']
    #pred = pd.Series(pred, index=test.index)
    #pred_gwn = pred_dict_gwn['median']
    #pred_gwn = pd.Series(pred_gwn, index=test.index)
    #pred_spsa = pred_dict_spsa['median']
    #pred_spsa = pd.Series(pred_spsa, index=test.index)
    #pred_memory = pred_dict_memory['median']
    #pred_memory = pd.Series(pred_memory, index=test.index)
    #model_name = model
    #plt.figure(figsize=(8, 6), dpi=100)
    #plt.plot(train)
    #plt.plot(test, label='Truth', color='black')
    #plt.plot(pred, label=model_name, color='purple')
    #plt.plot(pred_gwn, label='GWN', color ='yellow')
    #plt.plot(pred_spsa, label='SPSA', color = 'red')
    ##plt.plot(pred_memory, label='memory', color = 'orange')
    #plt.legend(loc='upper left')
    #if 'NLL/D' in pred_dict:
    #    nll = pred_dict['NLL/D']
    #    if nll is not None:
    #        plt.text(0.03, 0.85, f'NLL/D: {nll:.2f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    #plt.show()
    #plt.savefig('output_graphics/testattack_4.png')
