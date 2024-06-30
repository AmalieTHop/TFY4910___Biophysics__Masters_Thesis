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
from simulations.simulation_params_sup_b5 import simulation_params as simulation_params


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr



class train_pars:
    def __init__(self):
        self.optim ='adamw'
        self.lr = 0.004332
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

class net_pars:
    def __init__(self):
        self.dropout = 0.118
        self.batch_norm = True
        self.parallel = True
        self.con = 'sigmoid'
        self.bounds = np.array([[0.0005, 0.05, 0.005, 0.7], [0.003, 0.50, 0.1, 1.3]])
        self.fitS0 = True
        self.depth = 2
        self.width = 84

class hyperparams_selfsupervised:
    def __init__(self):
        self.net_pars = net_pars()
        self.train_pars = train_pars()
        self.norm_data_full = False
        self.id = f'5sup_optim_snr20_nmae_0509_d{self.net_pars.depth}_w{self.net_pars.width}_o{self.train_pars.optim}_l{self.train_pars.lr}_{self.train_pars.loss_fun}_{self.net_pars.con}_d{self.net_pars.dropout}'


########################################################



# simulate
def run_sims(snr):
    print(f'SNR: {snr}')

    arg_sim = simulation_params()
    arg_sim = from_simulations.checkarg_simulation_params(arg_sim)

    # load hyperparameter
    arg_dnn = hyperparams_selfsupervised()
    arg_dnn = from_DNN.checkarg(arg_dnn)

    # modify this
    save_name = f'r{arg_sim.repeats}_optim_snr20_d0509' 
    
    # make directory
    dir_out = f'../../../simulations/simulations_data/b{len(arg_sim.bvalues)}/snr{snr}/dnn_{arg_sim.learning}/{save_name}'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    print(dir_out)
    
    # save model name
    np.save(os.path.join(dir_out, f'{arg_dnn.id}'), np.array([0]))

    # run simulation
    from_simulations.sim_dnn(arg_sim, snr, arg_sim.learning, arg_dnn, dir_out)


run_sims(snr)
