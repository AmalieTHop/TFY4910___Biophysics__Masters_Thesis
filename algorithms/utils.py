
import numpy as np

def ivim(bvalues, Dt, Fp, Dp, S0):
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))


def ivims(bvals, Dt, Fp, Dp, S0, nums):
    Sb_nums = []
    for i in range(nums):
        Sb_num = ivim(bvals, Dt[i], Fp[i], Dp[i], S0[i])
        Sb_nums.append(np.array(Sb_num))
    return np.array(Sb_nums)


def rmse(computed, measured):
        rmse = np.sqrt(np.mean(np.square(computed - measured), axis=-1))
        return rmse


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


def unormalise_params_new(params_norm, bounds):
    params_unorm = np.zeros(params_norm.shape)
    for i in range(len(params_norm)):
        params_unorm[i] = bounds[0, i] + params_norm[i] * (bounds[1, i] - bounds[0, i])
    return params_unorm

def normalise_params_new(params_unorm, bounds):
    params_norm = np.zeros(params_unorm.shape)
    for i in range(len(params_unorm)):
        params_norm[i] = (params_unorm[i]  - bounds[0, i]) / (bounds[1, i] - bounds[0, i])
    return params_norm

def unormalise_params(params_norm, bounds):
    params_unorm = []
    for i, param_norm in enumerate(params_norm):
        param_unorm = unormalise_param(param_norm, bounds[0, i], bounds[1, i])
        params_unorm.append(param_unorm)
    return params_unorm