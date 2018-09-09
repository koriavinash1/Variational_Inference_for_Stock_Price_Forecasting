import os
import numpy as np
import torch
import torch.nn as nn

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam



class BayesianMLP(nn.Module):
    def __init__(self, p):
        # p = number of features
        super(BayesianMLP, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        return self.linear(x)



class RNN_Regressor(nn.Module):
	def __init__(self, p):
        # p = number of features
        super(RNN_Regressor, self).__init__()
        ## TODO: ...

    def forward(self, x):
        return 