# import
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import simulations.simulations as from_simulations
from simulations.simulation_params_slf_b4 import simulation_params as simulation_params
from algorithms.fitting_algos.fitting_params import lsq_params as lsq_params
from algorithms.fitting_algos.fitting_params import seg_params as seg_params

# argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--snr', dest='snr', type=int)
args = parser.parse_args()
snr = args.snr


# simulate
def run_sims(snr):
    print(f'SNR: {snr}')

    arg_sim = simulation_params()
    arg_sim = from_simulations.checkarg_simulation_params(arg_sim)

    arg_lsq = lsq_params()
    arg_seg = seg_params()

    arg_seg.cutoff = 100

    # make directory
    dir_out = f'../../../simulations/simulations_data'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # run simulation
    from_simulations.sim_fit(arg_sim, snr, arg_lsq, arg_seg, dir_out)


run_sims(snr)
