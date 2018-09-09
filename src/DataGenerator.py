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

# startDate = date(1997, 1, 1)
# endDate = date(2010, 1, 1)
# startDate = date(2010, 1, 1)

class DatasetGenerator(object):
    """docstring for DatasetGenerator"""
    def __init__(self, symbol = symbol, mode = 'train'):
        super(DatasetGenerator, self).__init__()

        data   = get_history(symbol = symbol,
                            start = startDate,
                            end = endDate)

        if mode   == 'train': self.data = data[:int(len(data)*train_per)]
        elif mode == 'valid': self.data = data[int(len(data)*train_per):int(len(data)*(train_per + valid_per))]
        elif mode == 'test':  self.data = data[int(len(data)*(train_per + valid_per)):]
        else : raise ValueError('Unknown mode, Allowed mode are: {}'.format(['train', 'valid', 'test']))

        # details: [Symbol, Series, Prev Close, Open, High, Low, Last, Close, VWAP, \
        #                       Volume, Turnover, Trades, Deliverable Volume, %Deliverble]
        
        self.data = data.as_matrix()
        
    
    def __getitem__(self, index):
        x_train = 0
        y_train = 0
        return x_train, y_train

    def __len__(self)
        return len(self.data)  



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