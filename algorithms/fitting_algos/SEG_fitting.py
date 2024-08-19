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
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""

from scipy.optimize import curve_fit
import numpy as np
import tqdm


import algorithms.utils as from_utils



def monofit(bvalues, Dt, Fp):
    return (1-Fp)*np.exp(-bvalues * Dt)





def fit_least_squares_array(bvalues, dw_data, mask_data = np.array([0]), bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]), fitS0=True,  cutoff=200):
    """
    This is the SEG implementation. In the first step, Dt is estimated using a curve fit to b-values>=cutoff. In
    the second step, the parameters fp, Dp and S0 are fitted using all b-values, with Dt being fixed from the first 
    step. This fit is done on an array.
    
    :param bvalues:     1D Array with the b-values
    :param dw_data:     2D Array with diffusion-weighted signal in different voxels at different b-values
    :param mask_data    Array with binary values representiing the ROI (e.g. GTVn).
    :param fitS0:       Boolean determining whether to fit S0 to 1; default = True. fitS0=False it not (yet) implemented.
    :param bounds:      Array with fitting boundaries ([Dt_min, Fp_min, Dp_min, S0_min],[Dt_max, Fp_max, Dp_max, S0_max]).
    :param cutoff:      Cut-off b-value

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
    Dt = np.zeros(len(dw_data))
    Fp = np.zeros(len(dw_data))
    Dp = np.zeros(len(dw_data))
    S0 = np.zeros(len(dw_data))
    rmse = np.zeros(len(dw_data))
    
     # fill arrays with fit results on a per voxel base:
    if not mask_data.any():         # in scilico
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], bounds=bounds, fitS0=fitS0, cutoff=cutoff)
            dw_data_fit = from_utils.ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
            rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    else:                           # in vivo
        for i in tqdm.tqdm(range(len(dw_data)), position=0, leave=True):
            if (mask_data[i] != 0):
                Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares(bvalues, dw_data[i, :], bounds=bounds, fitS0=fitS0, cutoff=cutoff)
                dw_data_fit = from_utils.ivim(bvalues, Dt[i], Fp[i], Dp[i], S0[i])
                rmse[i] = np.sqrt(np.mean(np.square(dw_data_fit - dw_data[i, :]), axis=-1))
    
    return [Dt, Fp, Dp, S0, rmse]





def fit_least_squares(bvalues, dw_data, bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]), fitS0=True, cutoff=200):
    """
    This is the SEG implementation. In the first step, Dt is estimated using a curve fit to b-values>=cutoff. In
    the second step, the parameters fp, Dp and S0 are fitted using all b-values, with Dt being fixed from the first 
    step. It fits a single curve.

    :param bvalues: 1D Array with the b-values
    :param dw_data: 1D Array with diffusion-weighted signal in different voxels at different b-values
    :param fitS0:   Boolean determining whether to fit S0 to 1; default = True. fitS0=False it not (yet) implemented.
    :param bounds:  Array with fitting boundaries ([Dt_min, Fp_min, Dp_min, S0_min],[Dt_max, Fp_max, Dp_max, S0_max]).
    :param cutoff:  Cut-off b-value
    
    :return Dt:     Scalar with Dt of the specific voxel
    :return Fp:     Scalar with fp of the specific voxel
    :return Dp:     Scalar with Dp of the specific voxel
    :return S0:     Scalar with S0 of the specific voxel
    """

    ### first step ###
    
    # b-values and correspodnding dwi data with b-value greater or equal to cut-off b-value
    high_bvalues = bvalues[bvalues >= cutoff]
    high_bvalues_dw_data = dw_data[bvalues >= cutoff]

    # fitting boundaries
    boundsmonoexp = ([bounds[0][0], bounds[0][1]],
                     [bounds[1][0], bounds[1][1]])

    # initial guesses
    p0 = [0.001, 0.1]

    # non-linear least squares to fit the function, monofit, to data defined by high_bvalues and high_bvalues_dw_data
    params, _ = curve_fit(monofit, high_bvalues, high_bvalues_dw_data, p0=p0, bounds=boundsmonoexp, x_scale = [0.001, 0.1], maxfev=10000)
    Dt = params[0]


    ### second step ###

    # fitting boundaries
    boundsupdated = ([bounds[0][1], bounds[0][2] ,bounds[0][3]],
                     [bounds[1][1], bounds[1][2] ,bounds[1][3]])   
    
    # initial guesses
    p0 = [0.1, 0.025, 1]

    # non-linear least squares to fit the function, two_exp, to data defined by bvalues and dw_data
    params, _ = curve_fit(lambda b, Fp, Dp, S0: from_utils.ivim(b, Dt, Fp, Dp, S0), bvalues, dw_data, p0=p0, bounds=boundsupdated, x_scale = [0.1, 0.01, 1], maxfev=10000)
    Fp, Dp, S0 = params[0], params[1] , params[2]
        
    return Dt, Fp, Dp, S0