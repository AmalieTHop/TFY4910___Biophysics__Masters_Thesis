
import numpy as np
import matplotlib.ticker
from matplotlib import pyplot as plt

import os

from scipy.stats import bootstrap



class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat='%1.1f', offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat


def ivim(bval, Dt, Fp, Dp, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bval * Dp) + (1 - Fp) * np.exp(-bval * Dt)))


def ivims(bvals, Dt, Fp, Dp, S0, nums):
    Sb_nums = []
    for i in range(nums):
        Sb_num = ivim(bvals, Dt[i], Fp[i], Dp[i], S0[i])
        Sb_nums.append(np.array(Sb_num))
    return np.array(Sb_nums)


def rmse(computed, measured):
        rmse = np.sqrt(np.mean(np.square(computed - measured), axis=-1))
        return rmse


def ivims_3d(bvals, Dt, Fp, Dp, S0):
    num_ivims = Dt.size
    Dt_1d, Fp_1d, Dp_1d, S0_1d = Dt.reshape(num_ivims), Fp.reshape(num_ivims), Dp.reshape(num_ivims), S0.reshape(num_ivims)

    ivims_1d = ivim(np.tile(np.expand_dims(bvals, axis=0), (num_ivims, 1)),
                          np.tile(np.expand_dims(Dt_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(Fp_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(Dp_1d, axis=1), (1, len(bvals))),
                          np.tile(np.expand_dims(S0_1d, axis=1), (1, len(bvals)))).astype('f')

    ivims = ivims_1d.reshape((Dt.shape[0], Dt.shape[1], Dt.shape[2], len(bvals)))
    return ivims



def plot_loss(dir_in, dir_out, end_name = '', save_name=''):
    params = {'axes.labelsize': 20,
            'axes.titlesize': 22.5,
            'lines.marker': 'None',
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'legend.fontsize': 17.5,
            'legend.loc':'upper right',
            'legend.framealpha': 0.75,
            'savefig.format': 'pdf'
            }
    plt.rcParams.update(params)

    training_loss = np.load(os.path.join(dir_in, f'loss_train{end_name}.npy'))
    validation_loss = np.load(os.path.join(dir_in, f'loss_val{end_name}.npy'))
    best_validation_loss = np.load(os.path.join(dir_in, f'loss_val_best{end_name}.npy'))
    iterations = np.arange(len(training_loss))

    
    fig, ax = plt.subplots(figsize = [8,6])
    ax.plot(iterations, training_loss, label = "training loss")
    ax.plot(iterations, validation_loss, label = "validation loss")
    ax.set(xlabel = r"loss")
    ax.set(ylabel = r"epoch #")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    plt.savefig(os.path.join(dir_out, f'training_and_validation_loss{save_name}'))




def bootstrap_CI_99_mean(data):
    bootstrap_results = bootstrap((data,), np.mean, n_resamples=1000, random_state=8)
    ci_l, ci_u = bootstrap_results.confidence_interval
    return ci_l, ci_u