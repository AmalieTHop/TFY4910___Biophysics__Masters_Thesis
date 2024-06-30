import torch
import numpy as np


class train_pars:
    def __init__(self):
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True


class net_pars:
    def __init__(self):
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])


class hyperparams:
    def __init__(self):
        self.net_pars = net_pars()
        self.train_pars = train_pars()
        self.norm_data_full = False