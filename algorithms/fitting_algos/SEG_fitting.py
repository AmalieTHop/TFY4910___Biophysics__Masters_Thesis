"""
January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

requirements:
numpy
tqdm
scipy
"""

"""
Modified:
June 2024 by Amalie Toftum Hop
"""

# load relevant libraries
from scipy.optimize import curve_fit
import numpy as np
import tqdm


import algorithms.utils as from_utils



def two_exp_noS0(bvalues, Dpar, Fmv, Dmv):
    """ bi-exponential IVIM function, and S0 set to 1"""
    return Fmv * np.exp(-bvalues * Dmv) + (1 - Fmv ) * np.exp(-bvalues * Dpar)
       
def two_exp(bvalues, S0, Dpar, Fmv, Dmv):
    """ bi-exponential IVIM function"""
    return S0 * (Fmv * np.exp(-bvalues * Dmv) + (1 - Fmv ) * np.exp(-bvalues * Dpar))
   
def ivim(bvalues, Dt, Fp, Dp, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))


def monofit(bvalues, Dpar, Fmv):
    return (1-Fmv)*np.exp(-bvalues * Dpar)


def fit_least_squares_array(bvalues, dw_data, mask_data = np.array([0]), bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]), fitS0=True,  cutoff=200):
    """
    This is the LSQ implementation, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
    Second, we fit the other parameters using all b-values, while fixing Dpar from step 1. This fit
    is done on an array.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fmvmax, Dmvmax]). 
    :param cutoff: cutoff b-value used in step 1 
    :return Dpar: 1D Array with Dpar in each voxel
    :return Fmv: 1D Array with Fmv in each voxel
    :return Dmv: 1D Array with Dmv in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # first we normalise the signal to S0
    S0 = np.squeeze(dw_data[:, bvalues == 0])
    dw_data = dw_data / S0[:, None]
    if mask_data.any():
        mask_data[S0==0] = 0

    # initialize empty arrays
    Dt = np.zeros(len(dw_data))
    Fp = np.zeros(len(dw_data))
    Dp = np.zeros(len(dw_data))
    S0 = np.zeros(len(dw_data))
    rmse = np.zeros(len(dw_data))
    
     # fill arrays with fit results on a per voxel base:
    if not mask_data.any():
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], bounds=bounds, fitS0=fitS0, cutoff=cutoff)
            dw_data_fit = ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
            rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    else:
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            if (mask_data[i] != 0):
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], bounds=bounds, fitS0=fitS0, cutoff=cutoff)
                dw_data_fit = from_utils.ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
                rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    return [Dt, Fp, Dp, S0, rmse]


def fit_least_squares(bvalues, dw_data, bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]), fitS0=True, cutoff=200):
    """
    This is the LSQ implementation, in which we first estimate Dpar using a curve fit to b-values>=cutoff;
    Second, we fit the other parameters using all b-values, while fixing Dpar from step 1. This fit
    is done on an array. It fits a single curve
    :param bvalues: 1D Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; 
    :param fitS0: Boolean determining whether to fix S0 to 1;
    :param bounds: Array with fit bounds ([S0min, Dparmin, Fmvmin, Dmvmin],[S0max, Dparmax, Fmvmax, Dmvmax]). 
    :param cutoff: cutoff b-value used in step 1 
    :return S0: optional 1D Array with S0 in each voxel
    :return Dpar: scalar with Dpar of the specific voxel
    :return Fmv: scalar with Fmv of the specific voxel
    :return Dmv: scalar with Dmv of the specific voxel
    """
    
    high_b = bvalues[bvalues >= cutoff]
    high_dw_data = dw_data[bvalues >= cutoff]

    # first step
    boundsmonoexp = ([bounds[0][0], bounds[0][1]],
                     [bounds[1][0], bounds[1][1]])

    p0 = [0.001, 0.1]
    params, _ = curve_fit(monofit, high_b, high_dw_data, p0=p0, bounds=boundsmonoexp, x_scale = [0.001, 0.1], maxfev=10000)
    Dpar1 = params[0]

    # second step
    boundsupdated = ([bounds[0][3] , bounds[0][1] , bounds[0][2]],
                     [bounds[1][3] , bounds[1][1] , bounds[1][2]])   
    p0 = [1, 0.1, 0.025]
    params, _ = curve_fit(lambda b, S0, Fmv, Dmv: two_exp(b, S0, Dpar1, Fmv, Dmv), bvalues, dw_data, p0=p0, bounds=boundsupdated, x_scale = [1, 0.1, 0.01], maxfev=10000)
    S0, Fmv, Dmv = params[0], params[1] , params[2]
        
    return Dpar1, Fmv, Dmv, S0
