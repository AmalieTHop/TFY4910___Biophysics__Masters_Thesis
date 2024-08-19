"""
September 2020 by Oliver Gurney-Champion
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Code is uploaded as part of our publication in MRM (Kaandorp et al. Improved physics-informed deep learning of the intravoxel-incoherent motion model: accurate, unique and consistent. MRM 2021)
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


import numpy as np
import time
import torch
import os
import warnings

import scipy.stats as scipy

import algorithms.DNN.DNN as from_DNN
import algorithms.fitting_algos.LSQ_fitting as from_LSQ_fitting
import algorithms.fitting_algos.SEG_fitting as from_SEG_fitting
import algorithms.utils as from_utils    





def sim_dnn(arg_sim, snr, learning, arg_dnn, dir_out):
    """
    In silico with the DNN algorithms.
    """
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dims = 4

    # training
    signal_noisy_training, _, params_train_norm = sim_signal(snr, arg_sim.bvalues, 
                                                             num_samples=arg_sim.num_samples_training, 
                                                             ranges=arg_sim.ranges,
                                                             rician=arg_sim.rician, 
                                                             state=16)
    
    # evalutation/test
    signal_noisy_test, params_test_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=18)

    
    # all output predictions for all repeated trainings are stored in 'params_dnn'.
    params_dnn = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])


    # loop over repeated trainings
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
        np.save(os.path.join(dir_out, f'loss_valES_{aa}'), loss_valES)
        np.save(os.path.join(dir_out, f'loss_valES_best_{aa}'), loss_valES_best)
        print('Trained model is saved')
        
        # load network: comment out the lines above where network is trained, uncomment this line, and fill in string to trained network
        # net = torch.load(f'fill_in_string')

        # predict parameters on test set
        start_time = time.time()
        params_dnn[aa], loss_test = from_DNN.predict_IVIM(signal_noisy_test, arg_sim.bvalues, net, arg_dnn)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        np.save(os.path.join(dir_out, f'loss_test_{aa}'), loss_test)


        # remove network to save memory
        del net
        if arg_dnn.train_pars.use_cuda:
            torch.cuda.empty_cache()
            

    # evaluation
    
    # compute errors and spearmans rho
    mat_dnn = np.zeros([arg_sim.repeats, dims-1, 4])
    for aa in range(arg_sim.repeats):
        mat_dnn[aa] = compute_errors_and_rho(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_dnn[aa])
    np.save(os.path.join(dir_out, f'mat_dnn_raw'), mat_dnn)
    mat_dnn = np.mean(mat_dnn, axis=0)
            
    # compute cv
    if arg_sim.repeats > 1:
        cv = np.sqrt(np.mean(np.square(np.std(params_dnn, axis=0)), axis=1))
        cv = cv[[0, 1, 2]] / [np.mean(params_test_unorm[0]), np.mean(params_test_unorm[1]), np.mean(params_test_unorm[2])]
    else:
        cv = np.zeros(dims-1)
        
    # save
    np.save(os.path.join(dir_out, f'params_dnn'), params_dnn)
    np.save(os.path.join(dir_out, f'mat_dnn'), mat_dnn)
    np.save(os.path.join(dir_out, f'cv'), cv)
            
    del params_dnn





def sim_fit(arg_sim, snr, arg_lsq, arg_seg, dir_out):
    """
    In silico with the fitting algorithms.
    """

    if not os.path.exists(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq')):
        os.makedirs(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq'))
    if not os.path.exists(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg')):
        os.makedirs(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg'))


    # evaluation/test
    signal_noisy_test, params_test_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=18)
    

    # compute and save signal-rmse between the true (noiseless) and the observed (noisy) signal sequences
    signal_noiseless_test = from_utils.ivims(arg_sim.bvalues, params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_test_unorm[3], arg_sim.num_samples_test)
    observed_signal_rmse_test_gt = from_utils.rmse(signal_noiseless_test, signal_noisy_test)
    np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/observed_signal_rmse_test_gt'), observed_signal_rmse_test_gt)

    # save gt ivim parameters and observed signal sequences
    np.save(os.path.join(dir_out, f'Dt_gt'), params_test_unorm[0])
    np.save(os.path.join(dir_out, f'Fp_gt'), params_test_unorm[1])
    np.save(os.path.join(dir_out, f'Dp_gt'), params_test_unorm[2])
    np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/signal_noisy_test'), signal_noisy_test) 


    # lsq
    if arg_lsq.do_fit:

        # fit
        start_time = time.time()
        params_lsq = from_LSQ_fitting.fit_least_squares_array(arg_sim.bvalues, signal_noisy_test, fitS0=arg_lsq.fitS0, bounds=arg_lsq.bounds)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for lsq fit: {elapsed_time}')

        # compute signal-rmse between the predicted and the observed (noisy) signal sequences
        signal_pred_lsq_test = from_utils.ivims(arg_sim.bvalues, params_lsq[0], params_lsq[1], params_lsq[2], params_lsq[3], arg_sim.num_samples_test)
        observed_signal_rmse_test_pred_lsq = from_utils.rmse(signal_pred_lsq_test, signal_noisy_test)
        
        # determine errors and Spearman Rank
        mat_lsq = compute_errors_and_rho(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_lsq)

        # save
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/observed_signal_rmse_test_pred'), observed_signal_rmse_test_pred_lsq)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/params_lsq'), params_lsq)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/lsq/mat_lsq'), mat_lsq)


    # seg
    if arg_seg.do_fit:

        #fit
        start_time = time.time()
        params_seg = from_SEG_fitting.fit_least_squares_array(arg_sim.bvalues, signal_noisy_test, bounds=arg_seg.bounds, fitS0=arg_seg.fitS0, cutoff=arg_seg.cutoff)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for lsq fit: {elapsed_time}')

        # determine signal-rmse between noisy and predicted
        signal_pred_seg_test = from_utils.ivims(arg_sim.bvalues, params_seg[0], params_seg[1], params_seg[2], params_seg[3], arg_sim.num_samples_test)
        observed_signal_rmse_test_pred_seg = from_utils.rmse(signal_pred_seg_test, signal_noisy_test)

        # determine errors and Spearman Rank
        mat_seg = compute_errors_and_rho(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_seg)

        # save
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/observed_signal_rmse_test_pred'), observed_signal_rmse_test_pred_seg)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/params_seg'), params_seg)
        np.save(os.path.join(dir_out, f'b{len(arg_sim.bvalues)}/snr{snr}/seg/mat_seg'), mat_seg)





def sim_signal(snr, bvalues, num_samples=100000, ranges=np.array([[0.0005, 0.05, 0.005], [0.003, 0.50, 0.1]]), rician=True, state=123):
    """
    Generates synthetic signal sequences given by SNR, b-values and parameter intervals with Rician noise (if rician=True).
    The number of synthethic signal sequences are given by num_samples. 
    """

    # randomly select parameters from predefined range
    rg = np.random.RandomState(state)
    Dt_norm = rg.uniform(0, 1, (num_samples))
    Fp_norm = rg.uniform(0, 1, (num_samples))
    Dp_norm = rg.uniform(0, 1, (num_samples))
    [Dt_unorm, Fp_unorm, Dp_unorm] = from_utils.unormalise_params(np.array([Dt_norm, Fp_norm, Dp_norm]), ranges)

    # initialise data array
    data_sim = np.zeros([num_samples, len(bvalues)])
    
    if snr == 0:
        addnoise = False
    else:
        addnoise = True
        
    
    # compute the synthetically generated IVIM signal sequences corresponding to the syntetically generated IVIM parameters
    for aa in range(num_samples):
        data_sim[aa, :] = from_utils.ivim(bvalues, Dt_unorm[aa], Fp_unorm[aa], Dp_unorm[aa], 1)

    # noise
    if addnoise:

        # initialise noise arrays
        noise_imag = np.zeros([num_samples, len(bvalues)])
        noise_real = np.zeros([num_samples, len(bvalues)])

        # generate real and imagniary noise
        for i in range(0, num_samples):
            noise_real[i,:] = rg.normal(0, 1 / snr, (1, len(bvalues)))
            noise_imag[i,:] = rg.normal(0, 1 / snr, (1, len(bvalues)))

        if rician:
            # add Rician noise: square root of squared gaussian distributed real signal + noise and imaginary noise
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # add Gaussian noise
            data_sim = data_sim + noise_imag


    # normalise to S(b=0)
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]

    # normalises S0 to be between 0 and 1 as this is needed for the supervised loss function
    S0_unorm = S0_noisy
    S0_norm = from_utils.normalise_param(S0_unorm, lower_bound=np.min(S0_unorm), upper_bound=np.max(S0_unorm))

    # normalised and scaled IVIM parameters
    params_unorm = np.array([Dt_unorm, Fp_unorm, Dp_unorm, S0_unorm])
    params_norm = np.array([Dt_norm, Fp_norm, Dp_norm, S0_norm])

    return data_sim, params_unorm, params_norm





def compute_errors_and_rho(Dt, Fp, Dp, params):
    """
    Computes NMAE and Spearmans rho (and NRMSE).
    """

    # norm
    normDt = np.mean(Dt)
    normFp = np.mean(Fp)
    normDp = np.mean(Dp)

    # rmse
    rmse_Dt = np.sqrt(np.square(np.subtract(Dt, params[0])).mean())
    rmse_Fp = np.sqrt(np.square(np.subtract(Fp, params[1])).mean())
    rmse_Dp = np.sqrt(np.square(np.subtract(Dp, params[2])).mean())

    # mae
    mae_Dt = np.mean(np.abs(np.subtract(Dt, params[0])))
    mae_Fp = np.mean(np.abs(np.subtract(Fp, params[1])))
    mae_Dp = np.mean(np.abs(np.subtract(Dp, params[2])))
    
    # rho and p-value
    Spearman = np.zeros([3, 2])
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params[0], params[2])  # Dt-Dp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params[0], params[1])  # Dt-fp
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params[1], params[2])  # fp-Dp
    Spearman[np.isnan(Spearman)] = 1                    # if rho is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman = np.absolute(Spearman)                    # absolute value
    

    # norm, nrmse, rho, nmae
    mats = [[normDt, rmse_Dt/normDt, Spearman[0, 0], mae_Dt/normDt],
            [normFp, rmse_Fp/normFp, Spearman[1, 0], mae_Fp/normFp],
            [normDp, rmse_Dp/normDp, Spearman[2, 0], mae_Dp/normDp]]

    del params
    return mats



def checkarg_simulation_params(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg_sim.bvalues not defined. Using default value of [0, 50, 100, 800]')
        arg.bvalues = np.array([0, 50, 100, 800])
    if not hasattr(arg, 'num_samples_training'):
        warnings.warn('arg_sim.num_samples_training not defined. Using default of 1000000')
        arg.num_samples_training = 100000
    if not hasattr(arg, 'num_samples_test'):
        warnings.warn('arg_sim.num_samples_test not defined. Using default of 100000')
        arg.num_samples_test = 100000
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg_sim.repeats not defined. Using default value of 25')
        arg.repeats = 25
    if not hasattr(arg, 'rician'):
        warnings.warn('arg_sim.rician not defined. Using default of True')
        arg.rician = True
    if not hasattr(arg, 'ranges'):
        warnings.warn('arg_sim.ranges not defined. Using default of ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])')
        arg.ranges = ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])
    if not hasattr(arg, 'learning'):
        warnings.warn('arg_sim.learning not defined. Using default of slf')
        arg.learning = 'slf'
    return arg