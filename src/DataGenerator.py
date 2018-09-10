from __future__ import division, print_function
import requests, json
requests.packages.urllib3.disable_warnings()
from nsepy import get_history
from nsetools import Nse

import os
import numpy as np
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from config import *
from max_vals import *

# startDate = date(1997, 1, 1)
# endDate = date(2010, 1, 1)
# startDate = date(2010, 1, 1)

def data_normalization(data, company, mode):
    """
    normalization technique followed: data.Object/ data.Object[0] -1
    max_values json array for all max values used in denormalizing the data
    """ 
    data.drop(data.columns[[5, 6, 2, 4, 3]], axis=1, inplace=True)
    # max_values = {'Open': data[1].astype("float")[0],\
    #             'Close': data[4].astype("float")[0],\
    #             'Low': data[3].astype("float")[0],\
    #             'High': data[2].astype("float")[0]}
    max_values = {'Close': data[1].astype("float")[0]}

    if IS_FIRST_RUN and mode == 'train':
        file = open("max_vals.py", "a")
        file.write(company + " = {NORM_VAL :" + str(data[1].astype("float")[0]) + ", MAX_VAL : " + str(max(data[1].astype("float"))) + '}')
        file.write("\n")
        file.close()
        data[1] = data[1].astype("float") / data[1].astype("float")[0] - 1
    else:
        data[1] = data[1].astype("float") / company['NORM_VAL']

    return data[1].as_matrix(), max_values


def construct_seq_data(array, sequence_length=sequence_length, prediction_length=prediction_length):
    """
    convert an array to sequencial data 
    one month data as one sequence
    """
    sequence_length = sequence_length + prediction_length
    return_data = np.array([ array[i: i+sequence_length] for i in range(array.shape[0]  - sequence_length)], ndmin=2)
    return return_data[:, :sequence_length - prediction_length], return_data[:, sequence_length - prediction_length:]


class DatasetGenerator(Dataset):
    """docstring for DatasetGenerator"""
    def __init__(self, symbol = symbol, mode = 'train', save = True):
        super(DatasetGenerator, self).__init__()

        data   = get_history(symbol = symbol,
                            start = startDate,
                            end = endDate)
        if save: data.to_csv(data_dir + symbol + '.csv')
        data = data.sample(frac=1).reset_index(drop=True)

        if mode   == 'train': self.data = data[:int(len(data)*train_per)]
        elif mode == 'valid': self.data = data[int(len(data)*train_per):int(len(data)*(train_per + valid_per))]
        elif mode == 'test':  self.data = data[int(len(data)*(train_per + valid_per)):]
        else : raise ValueError('Unknown mode, Allowed mode are: {}'.format(['train', 'valid', 'test']))

        # details: [Symbol, Series, Prev Close, Open, High, Low, Last, Close, VWAP, \
        #                       Volume, Turnover, Trades, Deliverable Volume, %Deliverble]
        data, _ = data_normalization(data, symbo, mode=mode)
        self.x_, self.y_ = construct_seq_data(data) 
        
    
    def __getitem__(self, index):
        x_train = torch.FloatTensor(self.x_[index])
        y_train = torch.FloatTensor(self.y_[index])
        return x_train, y_train

    def __len__(self)
        return len(self.x_)  



# test data...
def build_linear_dataset(N, p=1, noise_std=0.01):
    X = np.random.rand(N, p)
    # w = 3
    w = 3 * np.ones(p)
    # b = 1
    y = np.matmul(X, w) + np.repeat(1, N) + np.random.normal(0, noise_std, size=N)
    y = y.reshape(N, 1)
    X, y = torch.tensor(X).type(torch.Tensor), torch.tensor(y).type(torch.Tensor)
    data = torch.cat((X, y), 1)
    assert data.shape == (N, p + 1)
    return data