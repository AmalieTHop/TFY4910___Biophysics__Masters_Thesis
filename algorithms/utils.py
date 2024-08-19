"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import numpy as np
import torch
import torch.nn as nn



# IVIM model

def ivim(bvalues, Dt, Fp, Dp, S0):
    return S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt))


def ivims(bvals, Dt, Fp, Dp, S0, nums):
    Sb_nums = []
    for i in range(nums):
        Sb_num = ivim(bvals, Dt[i], Fp[i], Dp[i], S0[i])
        Sb_nums.append(np.array(Sb_num))
    return np.array(Sb_nums)



# RMSE

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction = 'none')
        self.eps = eps
        self.rmse = None
        
    def forward(self, yhat, y, reduction='mean'):
        se = self.mse(yhat,y)                            # sqared error
        self.rmse = torch.sqrt(torch.mean(se, dim=-1))   # rms error
        if reduction=='none':
            return self.rmse
        elif reduction=='mean':
            mean_rmse = torch.mean(self.rmse)            # mean rms error for batch
            return mean_rmse

def rmse(computed, measured):
        rmse = np.sqrt(np.mean(np.square(computed - measured), axis=-1))
        return rmse



# Normalisation and scaling

def normalise_param(param_unorm, lower_bound, upper_bound):
    param_norm = (param_unorm  - lower_bound) /(upper_bound - lower_bound)
    return param_norm

def normalise_params(params_unorm, bounds):
    params_norm = []
    for i, param_unorm in enumerate(params_unorm):
        param_norm = normalise_param(param_unorm, bounds[0, i], bounds[1, i])
        params_norm.append(param_norm)
    return params_norm

def unormalise_param(param_norm, lower_bound, upper_bound):
    param_unorm = lower_bound + param_norm * (upper_bound - lower_bound)
    return param_unorm

def unormalise_params(params_norm, bounds):
    params_unorm = []
    for i, param_norm in enumerate(params_norm):
        param_unorm = unormalise_param(param_norm, bounds[0, i], bounds[1, i])
        params_unorm.append(param_unorm)
    return params_unorm