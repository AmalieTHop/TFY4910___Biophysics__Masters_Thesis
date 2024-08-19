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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import bayesian_optimization.simulations_BO as from_simulations_BO
import bayesian_optimization.DNN_BO as from_DNN_BO
import bayesian_optimization.hyperparams_fixed_slf_BO as from_hyperparams_fixed_slf_BO
from bayesian_optimization.simulation_params_slf_b5_BO import simulation_params_slf_b5_BO as simulation_params_slf_b5_BO

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy



parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr



##############################################


def run_BO_sims(snr):
    print(f'SNR: {snr}')
    
    # load simulation parameters
    arg_sim = simulation_params_slf_b5_BO()
    arg_sim = from_simulations_BO.checkarg_simulation_params_BO(arg_sim)

    # load hyperparameter
    arg_dnn = from_hyperparams_fixed_slf_BO.hyperparams_slf_BO()
    arg_dnn = from_DNN_BO.checkarg_BO(arg_dnn)
    
    # make directory
    dir_out = f'../../../simulations/BO/b{len(arg_sim.bvalues)}/snr{snr}/dnn_{arg_sim.learning}'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    

    # specify strategy to chain multiple optimisation algorithms: first 25 experiments with SOBOL, then the rest with BOTORCH
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


    # define BO with hyperparameter search space 
    ax_client = AxClient(generation_strategy=gs, random_seed=4)
    ax_client.create_experiment(
        name="BO_slf_b5",  # The name of the experiment.
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
                "values": ['rmse'],
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
        objectives={
            "loss_valMS": ObjectiveProperties(minimize=True, threshold=0.030),
            "sum_nmae_valMS": ObjectiveProperties(minimize=True, threshold=1.0)},
        tracking_metric_names = ["loss_valES", "loss_valMS", "loss_test", "sum_nrmse_valMS", "sum_nrmse_test", "sum_nmae_valMS", "sum_nmae_test", "param_rmse_valMS", "param_rmse_test", "param_mae_valMS", "param_mae_test", "true_signal_rmse_valMS", "true_signal_rmse_test", "Dt_nrmse_valMS", "Fp_nrmse_valMS", "Dp_nrmse_valMS", "Dt_nrmse_test", "Fp_nrmse_test", "Dp_nrmse_test", "Dt_nmae_valMS", "Fp_nmae_valMS", "Dp_nmae_valMS", "Dt_nmae_test", "Fp_nmae_test", "Dp_nmae_test"]
    )


    
    """
    # Baseline network
    # Uncomment if comparison with specified a baseline network is desired. The BO will be dependent on this specified 
    # baseline network. The hyperparameters of the network used in the project thesis is added below for reference.
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
    ax_client.complete_trial(trial_index=0, raw_data=from_simulations_BO.sim_dnn_BO(arg_sim, snr, arg_sim.learning, arg_dnn, baseline_parameters, dir_out))
    """
    


    # run experiments and save all tracking metrics
    for i in range(arg_sim.num_trials):
        parameterization, trial_index = ax_client.get_next_trial()
        ax_client.complete_trial(trial_index=trial_index, raw_data=from_simulations_BO.sim_dnn_BO(arg_sim, snr, arg_sim.learning, arg_dnn, parameterization, dir_out))

        ax_client.save_to_json_file(os.path.join(dir_out, f'ax_client_snapshot_{i}.csv'))
        
        df_trials = ax_client.get_trials_data_frame()
        df_trials.to_csv(os.path.join(dir_out, f'df_trials_{i}.csv'), index=False)


run_BO_sims(snr)