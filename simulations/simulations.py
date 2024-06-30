"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)

requirements:
numpy
torch
tqdm
matplotlib
scipy
joblib
"""

"""
Modified:
June 2024 by Amalie Toftum Hop
"""


# import libraries
import algorithms.DNN.DNN as from_DNN

import algorithms.fitting_algos.LSQ_fitting as from_LSQ_fitting
import algorithms.fitting_algos.SEG_fitting as from_SEG_fitting
import algorithms.utils as from_utils    

import numpy as np
import time
import torch
import os
import warnings

import scipy.stats as scipy


def sim_dnn(arg_sim, snr, learning, arg_dnn, dir_out):
    torch.manual_seed(0)
    dims = 4

    # training
    signal_noisy_training, _, params_train_norm = sim_signal(snr, arg_sim.bvalues, 
                                                             num_samples=arg_sim.num_samples_training, 
                                                             ranges=arg_sim.ranges,
                                                             rician=arg_sim.rician, 
                                                             state=16)
    
    # evalutation
    signal_noisy_test, params_test_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=18)

    
    # prepare a larger array in case we repeat training
    params_dnn = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])


    # loop over repeats
    for aa in range(arg_sim.repeats):
        print(f'Repeat: {aa}')
        
        # train network
        start_time = time.time()
        if learning == 'sup':
            net, loss_train, loss_valES, loss_valES_best = from_DNN.learn_supervised(signal_noisy_training, params_train_norm.T, arg_sim.bvalues, arg_dnn)
        elif learning == 'slf':
            net, loss_train, loss_valES, loss_valES_best = from_DNN.learn_selfsupervised(signal_noisy_training, arg_sim.bvalues, arg_dnn)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for training: {elapsed_time}')

        # save trained network
        torch.save(net, os.path.join(dir_out, f'trained_model_net_{aa}'))
        np.save(os.path.join(dir_out, f'loss_train_{aa}'), loss_train)
        np.save(os.path.join(dir_out, f'loss_val_{aa}'), loss_valES)
        np.save(os.path.join(dir_out, f'loss_val_best_{aa}'), loss_valES_best)
        print('Trained model is saved')
        

        #net = torch.load(f'/Volumes/LaCie/TFY4910/dataA/output/test_projectsettingsslf_mae_relu6/EMIN_1064/dnn/trained_model_net')#torch.load(f'../dataA/simulations_test/rep5_linear/SNR{SNR}/trained_model_net_0')  

        # predict parameters on test set
        start_time = time.time()
        if arg_sim.repeats > 1:
            params_dnn[aa], loss_test = from_DNN.predict_IVIM(signal_noisy_test, arg_sim.bvalues, net, arg_dnn)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        np.save(os.path.join(dir_out, f'loss_test_{aa}'), loss_test)


        # remove network to save memory
        del net
        if arg_dnn.train_pars.use_cuda:
            torch.cuda.empty_cache()
            

    print('Results for dnn')
    # if we repeat training, then evaluate stability
    mat_dnn = np.zeros([arg_sim.repeats, dims-1, 4])
    
    # determine errors and Spearman Rank
    for aa in range(arg_sim.repeats):
        mat_dnn[aa] = print_errors(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_dnn[aa])
    np.save(os.path.join(dir_out, f'mat_dnn_raw'), mat_dnn)
    mat_dnn = np.mean(mat_dnn, axis=0)
            
    # calculate Stability Factor
    if arg_sim.repeats > 1:
        stability = np.sqrt(np.mean(np.square(np.std(params_dnn, axis=0)), axis=1))
        stability = stability[[0, 1, 2]] / [np.mean(params_test_unorm[0]), np.mean(params_test_unorm[1]), np.mean(params_test_unorm[2])]
    else:
        stability = np.zeros(dims-1)
        
    # save
    np.save(os.path.join(dir_out, f'params_dnn'), params_dnn)
    np.save(os.path.join(dir_out, f'mat_dnn'), mat_dnn)
    np.save(os.path.join(dir_out, f'stability'), stability)
            
    del params_dnn




def sim_fit(arg_sim, snr, arg_lsq, arg_seg, dir_out):
    if not os.path.exists(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq')):
        os.makedirs(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq'))
    if not os.path.exists(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg')):
        os.makedirs(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg'))

    signal_noisy_test, params_test_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=18)
    
    signal_nonoise_test = from_utils.ivims(arg_sim.bvalues, params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_test_unorm[3], arg_sim.num_samples_test)
    measured_signal_rmse_test_gt = from_utils.rmse(signal_nonoise_test, signal_noisy_test)
    np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/measured_signal_rmse_test_gt'), measured_signal_rmse_test_gt)

    # save
    np.save(os.path.join(dir_out, f'Dt_gt'), params_test_unorm[0])
    np.save(os.path.join(dir_out, f'Fp_gt'), params_test_unorm[1])
    np.save(os.path.join(dir_out, f'Dp_gt'), params_test_unorm[2])
    np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/signal_noisy_test'), signal_noisy_test) 

    
    if arg_lsq.do_fit:
        # fit
        start_time = time.time()
        params_lsq = from_LSQ_fitting.fit_least_squares_array(arg_sim.bvalues, signal_noisy_test, fitS0=arg_lsq.fitS0, bounds=arg_lsq.bounds)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for lsq fit: {elapsed_time}')

        # determine signal-rmse between noisy and predicted
        signal_pred_lsq_test = from_utils.ivims(arg_sim.bvalues, params_lsq[0], params_lsq[1], params_lsq[2], params_lsq[3], arg_sim.num_samples_test)
        measured_signal_rmse_test_pred_lsq = from_utils.rmse(signal_pred_lsq_test, signal_noisy_test)
        
        # determine errors and Spearman Rank
        print('Results for lsq fit')
        mat_lsq = print_errors(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_lsq)

        # save
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/measured_signal_rmse_test_pred'), measured_signal_rmse_test_pred_lsq)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/params_lsq'), params_lsq)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/mat_lsq'), mat_lsq)


    if arg_seg.do_fit:
        #fit
        start_time = time.time()
        params_seg = from_SEG_fitting.fit_least_squares_array(arg_sim.bvalues, signal_noisy_test, bounds=arg_seg.bounds, fitS0=arg_seg.fitS0, cutoff=arg_seg.cutoff)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for lsq fit: {elapsed_time}')

        # determine signal-rmse between noisy and predicted
        signal_pred_seg_test = from_utils.ivims(arg_sim.bvalues, params_seg[0], params_seg[1], params_seg[2], params_seg[3], arg_sim.num_samples_test)
        measured_signal_rmse_test_pred_seg = from_utils.rmse(signal_pred_seg_test, signal_noisy_test)

        # determine errors and Spearman Rank
        print('Results for seg fit')
        mat_seg = print_errors(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_seg)

        # save
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/measured_signal_rmse_test_pred'), measured_signal_rmse_test_pred_seg)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/params_seg'), params_seg)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/mat_seg'), mat_seg)



def sim_signal(snr, bvalues, num_samples=100000, ranges=np.array([[0.0005, 0.05, 0.005], [0.003, 0.5, 0.1]]), rician=True, state=123):
    # randomly select parameters from predefined range
    rg = np.random.RandomState(state)
    Dt_norm = rg.uniform(0, 1, (num_samples))
    Fp_norm = rg.uniform(0, 1, (num_samples))
    Dp_norm = rg.uniform(0, 1, (num_samples))
    [Dt_unorm, Fp_unorm, Dp_unorm] = from_utils.unormalise_params(np.array([Dt_norm, Fp_norm, Dp_norm]), ranges)

    # initialise data array
    data_sim = np.zeros([num_samples, len(bvalues)])
    bvalues = np.array(bvalues)
    
    if snr == 0:
        addnoise = False
    else:
        addnoise = True
        
    
    # loop over array to fill with simulated IVIM data
    for aa in range(num_samples):
        data_sim[aa, :] = from_utils.ivim(bvalues, Dt_unorm[aa], Fp_unorm[aa], Dp_unorm[aa], 1)

    # if snr is set to zero, don't add noise
    if addnoise:
        # initialise noise arrays
        noise_imag = np.zeros([num_samples, len(bvalues)])
        noise_real = np.zeros([num_samples, len(bvalues)])
        # fill arrays
        for i in range(0, num_samples):
            noise_real[i,:] = rg.normal(0, 1 / snr, (1, len(bvalues)))  # wrong! need a SD per input. Might need to loop to maD noise
            noise_imag[i,:] = rg.normal(0, 1 / snr, (1, len(bvalues)))
        if rician:
            # add Rician noise as the square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim

    # normalise signal
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]

    S0_unorm = S0_noisy
    S0_norm = from_utils.normalise_param(S0_unorm, lower_bound=np.min(S0_unorm), upper_bound=np.max(S0_unorm))

    params_unorm = np.array([Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm])
    params_norm = np.array([Dt_norm, Fp_norm, Dp_norm, S0_norm])

    return data_sim, params_unorm, params_norm



def print_errors(Dt, Fp, Dp, params):
    # this function calculates and prints the random, systematic, root-mean-squared (RMSE) errors and Spearman Rank correlation coefficient

    rmse_Dt = np.sqrt(np.square(np.subtract(Dt, params[0])).mean())
    rmse_Fp = np.sqrt(np.square(np.subtract(Fp, params[1])).mean())
    rmse_Dp = np.sqrt(np.square(np.subtract(Dp, params[2])).mean())

    mae_Dt = np.mean(np.abs(np.subtract(Dt, params[0])))
    mae_Fp = np.mean(np.abs(np.subtract(Fp, params[1])))
    mae_Dp = np.mean(np.abs(np.subtract(Dp, params[2])))
    
    # initialise Spearman Rank matrix
    Spearman = np.zeros([3, 2])
    # calculate Spearman Rank correlation coefficient and p-value
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params[0], params[2])  # DvDp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params[0], params[1])  # Dvf
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params[1], params[2])  # fvDp
    # If spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman[np.isnan(Spearman)] = 1
    # take absolute Spearman
    Spearman = np.absolute(Spearman)
    del params

    normDt = np.mean(Dt)
    normFp = np.mean(Fp)
    normDp = np.mean(Dp)

    mats = [[normDt, rmse_Dt/normDt, Spearman[0, 0], mae_Dt/normDt],
            [normFp, rmse_Fp/normFp, Spearman[1, 0], mae_Fp/normFp],
            [normDp, rmse_Dp/normDp, Spearman[2, 0], mae_Dp/normDp]]

    print(f'Results from NN: {mats}')

    return mats



def checkarg_simulation_params(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg_sim.bvalues not defined. Using default value of [0, 50, 100, 800]')
        arg.bvalues = np.array([0, 50, 100, 800])
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg_sim.repeats not defined. Using default value of 1')
        arg.repeats = 1  
    if not hasattr(arg, 'rician'):
        warnings.warn('arg_sim.rician not defined. Using default of True')
        arg.rician = True
    if not hasattr(arg, 'num_samples_training'):
        warnings.warn('arg_sim.num_samples_training not defined. Using default of 1000000')
        arg.num_samples_training = 1000000
    if not hasattr(arg, 'num_samples_test'):
        warnings.warn('arg_sim.num_samples_test not defined. Using default of 100000')
        arg.num_samples_test = 100000
    if not hasattr(arg, 'ranges'):
        warnings.warn('arg_sim.ranges not defined. Using default of ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])')
        arg.ranges = ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])
    return arg