import torch
import numpy as np


class train_pars:
    def __init__(self):
        self.optim ='adam'
        self.lr = 0.000015
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.load_nn= False
        self.loss_fun = 'rmse'
        self.skip_net = False
        self.scheduler = False
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True

class net_pars:
    def __init__(self):
        self.dropout = 0.177
        self.batch_norm = True
        self.parallel = True
        self.con = 'relu6'
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
        self.fitS0 = True
        self.depth = 2
        self.width = 70

class hyperparams_selfsupervised:
    def __init__(self):
        self.net_pars = net_pars()
        self.train_pars = train_pars()
        self.norm_data_full = False
        self.id = f'5slf_optim_snr20_nmaevalHT_0509_d{self.net_pars.depth}_w{self.net_pars.width}_o{self.train_pars.optim}_l{self.train_pars.lr}_{self.train_pars.loss_fun}_{self.net_pars.con}_d{self.net_pars.dropout}'
