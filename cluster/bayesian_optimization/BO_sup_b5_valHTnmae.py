# import
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import bayesian_optimization.simulations_BO as from_simulations_BO
import bayesian_optimization.DNN_BO as from_DNN_BO
from bayesian_optimization.simulation_params_sup_b5_BO import simulation_params as simulation_params

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import ModelRegistryBase, Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train



parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr



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
        self.bounds = np.array([[0.0005, 0.05, 0.005, 0.8], [0.003, 0.50, 0.1, 1.2]])


class hyperparams_BO:
    def __init__(self):
        self.net_pars = net_pars()
        self.train_pars = train_pars()
        self.norm_data_full = False



##############################################


# simulate
def run_sims(snr):
    print(f'SNR: {snr}')

    arg_sim = simulation_params()
    arg_sim = from_simulations_BO.checkarg_simulation_params(arg_sim)

    # load hyperparameter
    arg_dnn = hyperparams_BO()
    arg_dnn = from_DNN_BO.checkarg(arg_dnn)
    
    # make directory
    dir_out = f'../../../simulations/BO_valHTnmae_CPUQ_0507/b{len(arg_sim.bvalues)}/snr{snr}/dnn_{arg_sim.learning}'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=25,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
            ),
        ]
    )

    ax_client = AxClient(generation_strategy=gs, random_seed=4)
    ax_client.create_experiment(
        name="BO_sup_b5_valHTnmae",  # The name of the experiment.
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "bounds": [0.00001, 0.005],
                "value_type": "float",
            },
            {
                "name": "optimizer",  
                "type": "choice",  
                "values": ['adam', 'adamw'], 
                "value_type": "str"
            },
            {
                "name": "width",
                "type": "range",
                "bounds": [4, 128], 
                "value_type": "int"
            },
            {
                "name": "depth",
                "type": "choice",
                "values": [2, 4],
                "value_type": "int"
            },
            {
                "name": "loss_fun",
                "type": "choice",
                "values": ['mae', 'rmse', 'mse'],
                "value_type": "str"
            },
            {
                "name": "constraint",
                "type": "choice",
                "values": ['sigmoid', 'relu6'],
                "value_type": "str"
            },
            {
                "name": "dropout_p",
                "type": "range",
                "bounds": [0, 0.5],
                "value_type": "float"
            },

        ],
        objectives={"sum_nmae_valHT": ObjectiveProperties(minimize=True, threshold=1.0)},
        tracking_metric_names = ["loss_valES", "loss_test", "sum_nrmse_test", "sum_nmae_test", "param_rmse_valHT", "param_rmse_test", "param_mae_valHT", "param_mae_test", "true_signal_rmse_valHT", "true_signal_rmse_test", "Dt_nrmse_valHT", "Fp_nrmse_valHT", "Dp_nrmse_valHT", "Dt_nrmse_test", "Fp_nrmse_test", "Dp_nrmse_test", "Dt_nmae_valHT", "Fp_nmae_valHT", "Dp_nmae_valHT", "Dt_nmae_test", "Fp_nmae_test", "Dp_nmae_test"]

    )

    
    
    # baseline
    ax_client.attach_trial(
        parameters={"lr": 0.00003,
                    "optimizer": 'adam', 
                    "width": len(arg_sim.bvalues),
                    "depth": 2,
                    "loss_fun": 'rmse',
                    "constraint": 'sigmoid', 
                    "dropout_p": 0.1}
    )
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(trial_index=0, raw_data=from_simulations_BO.sim_dnn(arg_sim, snr, arg_sim.learning, arg_dnn, baseline_parameters, dir_out))
    

    

    for i in range(arg_sim.num_trials):
        parameterization, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=from_simulations_BO.sim_dnn(arg_sim, snr, arg_sim.learning, arg_dnn, parameterization, dir_out))

        ax_client.save_to_json_file(os.path.join(dir_out, f'ax_client_snapshot_{i}.csv'))

        df_trials = ax_client.get_trials_data_frame()
        df_trials.to_csv(os.path.join(dir_out, f'df_trials_{i}.csv'), index=False)


run_sims(snr)


