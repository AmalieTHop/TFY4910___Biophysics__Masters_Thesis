"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion
"""

"""
Modified:
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""

from scipy.optimize import curve_fit
import numpy as np
import sys
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable


import algorithms.fitting_algos.SEG_fitting as from_SEG_fitting
import algorithms.utils as from_utils



def ivimN(bvalues, Dt, Fp, Dp, S0):
    # IVIM function in which we try to have equal variance in the different IVIM parameters; equal variance helps with certain fitting algorithms
    return S0 * ivimN_noS0(bvalues, Dt, Fp, Dp)


def ivimN_noS0(bvalues, Dt, Fp, Dp):
    # IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


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
    
    :param bvalues:     1D Array with the b-values
    :param dw_data:     2D Array with diffusion-weighted signal in different voxels at different b-values
    :param mask_data    Array with binary values representiing the ROI (e.g. GTVn)
    :param fitS0:       Boolean determining whether to fit S0 to 1; default = True. fitS0=False it not (yet) implemented.
    :param bounds:      Array with fitting boundaries ([Dt_min, Fp_min, Dp_min, S0_min],[Dt_max, Fp_max, Dp_max, S0_max]).

    :return Dt:         1D Array with Dt in each voxel
    :return Fp:         1D Array with fp in each voxel
    :return Dp:         1D Array with Dp in each voxel
    :return S0:         1D Array with S0 in each voxel
    """

    # normalise to S(b=0)
    S0 = np.squeeze(dw_data[:, bvalues == 0])
    dw_data = dw_data / S0[:, None]

    # if in vivo, filter out voxels where S0 is zero
    if mask_data.any():
        mask_data[S0==0] = 0

    # initialize empty arrays
    Dp = np.zeros(len(dw_data))
    Dt = np.zeros(len(dw_data))
    Fp = np.zeros(len(dw_data))
    S0 = np.zeros(len(dw_data))
    rmse = np.zeros(len(dw_data))

    # fill arrays with fit results on a per voxel base
    if not mask_data.any():     # in scilico
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], fitS0=fitS0, bounds=bounds)
            dw_data_fit = from_utils.ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
            rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    else:                       # in vivo
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            if (mask_data[i] != 0):
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], fitS0=fitS0, bounds=bounds)
                dw_data_fit = from_utils.ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
                rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))

    return [Dt, Fp, Dp, S0, rmse]





def fit_least_squares(bvalues, dw_data, fitS0=True, bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.3, 1.3])): 
    """
    This is an implementation of the conventional IVIM fit. It fits a single curve.

    :param bvalues: 1D array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal in different voxels at different b-values
    :param fitS0:  Boolean determining whether to fit S0 to 1; default = True. fitS0=False it not (yet) implemented.
    :param bounds:  Array with fitting boundaries ([Dt_min, Fp_min, Dp_min, S0_min],[Dt_max, Fp_max, Dp_max, S0_max]).
    
    :return Dt:     Scalar with Dt of the specific voxel
    :return Fp:     Scalar with fp of the specific voxel
    :return Dp:     Scalar with Dp of the specific voxel
    :return S0:     Scalar with S0 of the specific voxel
    """
    try:
        # scale fitting boundaries
        bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
                  [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])
        
        #initial guesses
        p0=[1, 1, 0.1, 1]

        # non-linear least squares to fit the function, ivimN, to data defined by bvalues and dw_data
        params, _ = curve_fit(ivimN, bvalues, dw_data, p0=p0, bounds=bounds,  method='trf', maxfev=50000)

        # S0
        S0 = params[3]

        # correct for the rescaling of parameters
        Dt, Fp, Dp = params[0] / 1000, params[1] / 10, params[2] / 10

        # reorder output in case Dp < Dt
        return order(Dt, Fp, Dp, S0)
    except:
        # if fit fails, then do a segmented fit instead
        print('LSQ fit failed, trying SEG')
        Dt, Fp, Dp, S0 = from_SEG_fitting.fit_least_squares(bvalues, dw_data, bounds=bounds, fitS0=fitS0)
        return Dt, Fp, Dp, S0