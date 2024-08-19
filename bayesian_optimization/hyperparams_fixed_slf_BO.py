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



class train_pars_slf_BO:
    def __init__(self):
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True


class net_pars_slf_BO:
    def __init__(self):
        self.bounds = np.array([[0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]])


class hyperparams_slf_BO:
    def __init__(self):
        self.net_pars = net_pars_slf_BO()
        self.train_pars = train_pars_slf_BO()
        self.norm_data_full = False