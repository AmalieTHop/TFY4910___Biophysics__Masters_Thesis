"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



# import
import argparse
import os
import sys
current_dir = os.path.dirname(__file__)
grandparent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(grandparent_dir)
import numpy as np
import torch


import simulations.simulations as from_simulations
import algorithms.DNN.DNN as from_DNN
from simulations.simulation_params_sup_b4 import simulation_params_sup_b4 as simulation_params_sup_b4


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr



class train_pars_sup_b4:
    def __init__(self):
        self.optim ='adamw'
        self.lr = 0.00001
        self.patience = 10
        self.batch_size = 128
        self.maxit = 500
        self.split = 0.9
        self.load_nn= False
        self.loss_fun = 'mae'
        self.skip_net = False
        self.scheduler = False
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = True

class net_pars_sup_b4:
    def __init__(self):
        self.dropout = 0.092
        self.batch_norm = True
        self.parallel = True
        self.con = 'sigmoid'
        self.bounds = np.array([[0.0005, 0.05, 0.005, 0.7], [0.003, 0.50, 0.1, 1.3]])
        self.fitS0 = True
        self.depth = 4
        self.width = 101

class hyperparams_sup_b4:
    def __init__(self):
        self.net_pars = net_pars_sup_b4()
        self.train_pars = train_pars_sup_b4()
        self.norm_data_full = False
        self.id = f'4sup_optim_snr20_nmaevalMS_d{self.net_pars.depth}_w{self.net_pars.width}_o{self.train_pars.optim}_l{self.train_pars.lr}_{self.train_pars.loss_fun}_{self.net_pars.con}_d{self.net_pars.dropout}'

########################################################



# simulate
def run_sims(snr):
    print(f'SNR: {snr}')

    arg_sim = simulation_params_sup_b4()
    arg_sim = from_simulations.checkarg_simulation_params(arg_sim)

    # load hyperparameter
    arg_dnn = hyperparams_sup_b4()
    arg_dnn = from_DNN.checkarg(arg_dnn)

    # modify this
    save_name = f'r{arg_sim.repeats}_optim_snr20'
    
    # make directory
    dir_out = f'../../../simulations/simulations_data/b{len(arg_sim.bvalues)}/snr{snr}/dnn_{arg_sim.learning}/{save_name}'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    print(dir_out)
    
    # save model name
    np.save(os.path.join(dir_out, f'{arg_dnn.id}'), np.array([0]))

    # run simulations
    from_simulations.sim_dnn(arg_sim, snr, arg_sim.learning, arg_dnn, dir_out)


run_sims(snr)
