"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

requirements:
numpy
tqdm
matplotlib
scipy
joblib
"""

"""
Modified:
June 2024 by Amalie Toftum Hop
"""


# load relevant libraries
from scipy.optimize import curve_fit, minimize
import numpy as np
from scipy import stats
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable
import warnings


import algorithms.fitting_algos.SEG_fitting as from_SEG_fitting


def ivimN(bvalues, Dt, Fp, Dp, S0):
    # IVIM function in which we try to have equal variance in the different IVIM parameters; equal variance helps with certain fitting algorithms
    return S0 * ivimN_noS0(bvalues, Dt, Fp, Dp)


def ivimN_noS0(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def ivim(bvalues, Dt, Fp, Dp, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))


def order(Dt, Fp, Dp, S0=None):
    # function to reorder D* and D in case they were swapped during unconstraint fitting. Forces D* > D (Dp>Dt)
    if Dp < Dt:
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
    if S0 is None:
        return Dt, Fp, Dp
    else:
        return Dt, Fp, Dp, S0



def fit_least_squares_array(bvalues, dw_data, mask_data = np.array([0]), fitS0=True, 
                            bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])):
    """
    This is an implementation of the conventional IVIM fit. It is fitted in array form.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """

    # normalise the data to S(value=0)
    S0 = np.squeeze(dw_data[:, bvalues == 0]) ### np.mean(dw_data[:, bvalues == 0], axis=1)
    dw_data = dw_data / S0[:, None]
    if mask_data.any():
        mask_data[S0==0] = 0

    # defining empty arrays
    Dp = np.zeros(len(dw_data))
    Dt = np.zeros(len(dw_data))
    Fp = np.zeros(len(dw_data))
    S0 = np.zeros(len(dw_data))
    rmse = np.zeros(len(dw_data))

    # running in a single loop and filling arrays
    if not mask_data.any():
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], fitS0=fitS0, bounds=bounds)
            dw_data_fit = ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
            rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    else:
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            if (mask_data[i] != 0):
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], fitS0=fitS0, bounds=bounds)
                dw_data_fit = ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
                rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    return [Dt, Fp, Dp, S0, rmse]



def fit_least_squares(bvalues, dw_data, fitS0=True, bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])): 
    """
    This is an implementation of the conventional IVIM fit. It fits a single curve
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fix_S0: Boolean determining whether to fix S0 to 1; default = False
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: Array with D in each voxel
    :return Fp: Array with f in each voxel
    :return Dp: Array with Dp in each voxel
    :return S0: Array with S0 in each voxel
    """
    try:
        bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 100, bounds[0][3]],
                  [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 100, bounds[1][3]])
        p0=[1, 1, 2.5, 1]
        params, _ = curve_fit(ivimN, bvalues, dw_data, p0=p0, bounds=bounds,  method='trf', maxfev=50000)
        S0 = params[3]
        # correct for the rescaling of parameters
        Dt, Fp, Dp = params[0] / 1000, params[1] / 10, params[2] / 100
        # reorder output in case Dp<Dt
        return order(Dt, Fp, Dp, S0)
    except:
        # if fit fails, then do a segmented fit instead
        print('lsq fit failed, trying segmented')
        Dt, Fp, Dp, S0 = from_SEG_fitting.fit_least_squares(bvalues, dw_data, bounds=bounds, fitS0=fitS0)
        return Dt, Fp, Dp, S0


