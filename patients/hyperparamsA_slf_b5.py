"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import torch
import numpy as np


class train_pars_slf_b5:
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

class net_pars_slf_b5:
    def __init__(self):
        self.dropout = 0.177
        self.batch_norm = True
        self.parallel = True
        self.con = 'relu6'
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])
        self.fitS0 = True
        self.depth = 2
        self.width = 70

class hyperparams_slf_b5:
    def __init__(self):
        self.net_pars = net_pars_slf_b5()
        self.train_pars = train_pars_slf_b5()
        self.norm_data_full = False
        self.id = f'5slf_optim_snr20_nmaevalMS_d{self.net_pars.depth}_w{self.net_pars.width}_o{self.train_pars.optim}_l{self.train_pars.lr}_{self.train_pars.loss_fun}_{self.net_pars.con}_d{self.net_pars.dropout}'
