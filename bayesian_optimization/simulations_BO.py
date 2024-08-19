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



Comment: This source code is similar to simulations/simulations.py, but is adapted
for model selection using bayesian optimisation as described in the thesis. 
The parameter 'parameters' is used to handle the different hyperparameters 
that are being tuned.

Comment: This program is more extensive than what described in the thesis in the sense that is 
tracks more metrics. This is just provide extra information. It does not affect the model selection 
itself, as the model is selected manually based its preformance in terms of the metrics that are 
decribed in the theis.
"""


import numpy as np
import time
import torch
import warnings

import scipy.stats as scipy

import bayesian_optimization.DNN_BO as from_DNN_BO
import algorithms.DNN.DNN as from_DNN
import algorithms.utils as from_utils    
import simulations.simulations as from_simulations





def sim_dnn_BO(arg_sim, snr, learning, arg_dnn, parameters, dir_out):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    dims = 4

    # training
    signal_noisy_training, _, params_train_norm = from_simulations.sim_signal(snr, arg_sim.bvalues, 
                                                                              num_samples=arg_sim.num_samples_training, 
                                                                              ranges=arg_sim.ranges,
                                                                              rician=arg_sim.rician, 
                                                                              state=16)
    
    # validation
    signal_noisy_valMS, params_valMS_unorm, _ = from_simulations.sim_signal(snr, arg_sim.bvalues, 
                                                                            num_samples=arg_sim.num_samples_test, 
                                                                            ranges=arg_sim.ranges,
                                                                            rician=arg_sim.rician, 
                                                                            state=4)

    # evaluation/test
    signal_noisy_test, params_test_unorm, _ = from_simulations.sim_signal(snr, arg_sim.bvalues, 
                                                                          num_samples=arg_sim.num_samples_test, 
                                                                          ranges=arg_sim.ranges,
                                                                          rician=arg_sim.rician, 
                                                                          state=18)


    
    # all output predictions for all repeated trainings on validMS and test datasets are stored in 'params_dnn_valMS' and 'params_dnn_test', respectively.
    params_dnn_valMS = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])
    params_dnn_test = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])

    # losses for validES, validMS and test datasets for all repreated training are stored in the arrays below
    losses_valES = np.zeros([arg_sim.repeats])
    losses_valMS = np.zeros([arg_sim.repeats])
    losses_test = np.zeros([arg_sim.repeats])


    # loop over repeated trainings
    for aa in range(arg_sim.repeats):
        print(f'Repeat: {aa}')
        
        # train network
        start_time = time.time()
        if learning == 'sup':
            net, loss_train, loss_valES, losses_valES[aa] = from_DNN_BO.learn_supervised(signal_noisy_training, params_train_norm.T, arg_sim.bvalues, arg_dnn, parameters)
        elif learning == 'slf':
            net, loss_train, loss_valES, losses_valES[aa] = from_DNN_BO.learn_selfsupervised(signal_noisy_training, arg_sim.bvalues, arg_dnn, parameters)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for training: {elapsed_time}')

        """
        # save trained network
        #torch.save(net, os.path.join(dir_out, f'trained_model_net_{aa}'))
        #np.save(os.path.join(dir_out, f'loss_train_{aa}'), loss_train)
        #np.save(os.path.join(dir_out, f'loss_valES_{aa}'), loss_valES)
        #np.save(os.path.join(dir_out, f'loss_valES_best_{aa}'), losses_valES[aa])
        #print('Trained model is saved')
        """
        
        # load network: comment out the lines above where network is trained, uncomment this line, and fill in string to trained network
        # net = torch.load(f'fill_in_string')

        # predict parameters on validMS set
        start_time = time.time()
        params_dnn_valMS[aa], losses_valMS[aa] = from_DNN.predict_IVIM(signal_noisy_valMS, arg_sim.bvalues, net, arg_dnn)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        #np.save(os.path.join(dir_out, f'loss_valMS_{aa}'), losses_valMS[aa])

        # predict parameters on test set
        start_time = time.time()
        params_dnn_test[aa], losses_test[aa] = from_DNN.predict_IVIM(signal_noisy_test, arg_sim.bvalues, net, arg_dnn)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        #np.save(os.path.join(dir_out, f'loss_test_{aa}'), losses_test[aa])

        # remove network to save memory
        del net
        if arg_dnn.train_pars.use_cuda:
            torch.cuda.empty_cache()
            

    # evaluation

    # compute errors and spearmans rho
    mat_dnn_valMS = np.zeros([arg_sim.repeats, dims-1, 4])
    mat_dnn_test = np.zeros([arg_sim.repeats, dims-1, 4])
    param_rmse_valMS = np.zeros([arg_sim.repeats])
    param_rmse_test = np.zeros([arg_sim.repeats])
    param_mae_valMS = np.zeros([arg_sim.repeats])
    param_mae_test = np.zeros([arg_sim.repeats])
    mean_true_signal_rmse_pred_valMS = np.zeros([arg_sim.repeats])
    mean_true_signal_rmse_pred_test = np.zeros([arg_sim.repeats])
    for aa in range(arg_sim.repeats):
        mat_dnn_valMS[aa], param_rmse_valMS[aa], param_mae_valMS[aa], mean_true_signal_rmse_pred_valMS[aa] = compute_errors_and_rho_BO(params_valMS_unorm[0], params_valMS_unorm[1], params_valMS_unorm[2], params_dnn_valMS[aa], arg_sim.bvalues, arg_sim.num_samples_test)
        mat_dnn_test[aa], param_rmse_test[aa], param_mae_test[aa], mean_true_signal_rmse_pred_test[aa] = compute_errors_and_rho_BO(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_dnn_test[aa], arg_sim.bvalues, arg_sim.num_samples_test)

    # compute cv
    if arg_sim.repeats > 1:
        cv_valMS = np.sqrt(np.mean(np.square(np.std(params_dnn_valMS, axis=0)), axis=1))
        cv_test = np.sqrt(np.mean(np.square(np.std(params_dnn_valMS, axis=0)), axis=1))
        cv_valMS = cv_valMS[[0, 1, 2]] / [np.mean(params_valMS_unorm[0]), np.mean(params_valMS_unorm[1]), np.mean(params_valMS_unorm[2])]
        cv_test = cv_test[[0, 1, 2]] / [np.mean(params_test_unorm[0]), np.mean(params_test_unorm[1]), np.mean(params_test_unorm[2])]
    else:
        cv_valMS = np.zeros(dims-1)
        cv_test = np.zeros(dims-1)
        
    """
    # save
    #np.save(os.path.join(dir_out, f'params_dnn_valMS'), params_dnn_valMS)
    #np.save(os.path.join(dir_out, f'params_dnn_test'), params_dnn_test)
    #np.save(os.path.join(dir_out, f'mat_dnn_valMS'), mat_dnn_valMS)
    #np.save(os.path.join(dir_out, f'mat_dnn_test'), mat_dnn_test)
    #np.save(os.path.join(dir_out, f'param_rmse_valMS'), param_rmse_valMS)
    #np.save(os.path.join(dir_out, f'param_rmse_test'), param_rmse_test)
    #np.save(os.path.join(dir_out, f'param_mae_valMS'), param_mae_valMS)
    #np.save(os.path.join(dir_out, f'param_mae_test'), param_mae_test)
    #np.save(os.path.join(dir_out, f'mean_true_signal_rmse_pred_valMS'), mean_true_signal_rmse_pred_valMS)
    #np.save(os.path.join(dir_out, f'mean_true_signal_rmse_pred_test'), mean_true_signal_rmse_pred_test)
    #np.save(os.path.join(dir_out, f'cv_valMS'), cv_valMS)
    #np.save(os.path.join(dir_out, f'cv_test'), cv_test)
    """

    del params_dnn_valMS
    del params_dnn_test


    sums_nrsme_valMS = mat_dnn_valMS[:,0,1] + mat_dnn_valMS[:,1,1] + mat_dnn_valMS[:,2,1]
    mean_sum_nrsme_valMS = np.mean(sums_nrsme_valMS)
    std_sum_nrmse_valMS = np.std(sums_nrsme_valMS, ddof=1)

    sums_nrsme_test = mat_dnn_test[:,0,1] + mat_dnn_test[:,1,1] + mat_dnn_test[:,2,1]
    mean_sum_nrsme_test = np.mean(sums_nrsme_test)
    std_sum_nrmse_test = np.std(sums_nrsme_test, ddof=1)

    sums_nmae_valMS = mat_dnn_valMS[:,0,3] + mat_dnn_valMS[:,1,3] + mat_dnn_valMS[:,2,3]
    mean_sum_nmae_valMS = np.mean(sums_nmae_valMS)
    std_sum_nmae_valMS = np.std(sums_nmae_valMS, ddof=1)

    sums_nmae_test = mat_dnn_test[:,0,3] + mat_dnn_test[:,1,3] + mat_dnn_test[:,2,3]
    mean_sum_nmae_test = np.mean(sums_nmae_test)
    std_sum_nmae_test = np.std(sums_nmae_test, ddof=1)

    results_dict = {"loss_valES": (np.mean(losses_valES), np.std(losses_valES, ddof=1)), 
                    "loss_valMS": (np.mean(losses_valMS), np.std(losses_valMS, ddof=1)), 
                    "loss_test": (np.mean(losses_test), np.std(losses_test, ddof=1)), 
                    
                    "sum_nrmse_valMS": (mean_sum_nrsme_valMS, std_sum_nrmse_valMS),
                    "sum_nrmse_test": (mean_sum_nrsme_test, std_sum_nrmse_test),
                    
                    "sum_nmae_valMS": (mean_sum_nmae_valMS, std_sum_nmae_valMS),
                    "sum_nmae_test": (mean_sum_nmae_test, std_sum_nmae_test),
                    
                    "param_rmse_valMS": (np.mean(param_rmse_valMS), np.std(param_rmse_valMS, ddof=1)),
                    "param_rmse_test": (np.mean(param_rmse_test), np.std(param_rmse_test, ddof=1)),
                    
                    "param_mae_valMS": (np.mean(param_mae_valMS), np.std(param_mae_valMS, ddof=1)),
                    "param_mae_test": (np.mean(param_mae_test), np.std(param_mae_test, ddof=1)),
                    
                    "true_signal_rmse_valMS": (np.mean(mean_true_signal_rmse_pred_valMS), np.std(mean_true_signal_rmse_pred_valMS, ddof=1)),
                    "true_signal_rmse_test": (np.mean(mean_true_signal_rmse_pred_test), np.std(mean_true_signal_rmse_pred_test, ddof=1)),
                    
                    "Dt_nrmse_valMS": (np.mean(mat_dnn_valMS[:,0,1]), np.std(mat_dnn_valMS[:,0,1], ddof=1)), 
                    "Fp_nrmse_valMS": (np.mean(mat_dnn_valMS[:,1,1]), np.std(mat_dnn_valMS[:,1,1], ddof=1)),
                    "Dp_nrmse_valMS": (np.mean(mat_dnn_valMS[:,2,1]), np.std(mat_dnn_valMS[:,2,1], ddof=1)),
                    
                    "Dt_nrmse_test": (np.mean(mat_dnn_test[:,0,1]), np.std(mat_dnn_test[:,0,1], ddof=1)), 
                    "Fp_nrmse_test": (np.mean(mat_dnn_test[:,1,1]), np.std(mat_dnn_test[:,1,1], ddof=1)),
                    "Dp_nrmse_test": (np.mean(mat_dnn_test[:,2,1]), np.std(mat_dnn_test[:,2,1], ddof=1)),
                    
                    "Dt_nmae_valMS": (np.mean(mat_dnn_valMS[:,0,3]), np.std(mat_dnn_valMS[:,0,3], ddof=1)), 
                    "Fp_nmae_valMS": (np.mean(mat_dnn_valMS[:,1,3]), np.std(mat_dnn_valMS[:,1,3], ddof=1)),
                    "Dp_nmae_valMS": (np.mean(mat_dnn_valMS[:,2,3]), np.std(mat_dnn_valMS[:,2,3], ddof=1)),
                    
                    "Dt_nmae_test": (np.mean(mat_dnn_test[:,0,3]), np.std(mat_dnn_test[:,0,3], ddof=1)), 
                    "Fp_nmae_test": (np.mean(mat_dnn_test[:,1,3]), np.std(mat_dnn_test[:,1,3], ddof=1)),
                    "Dp_nmae_test": (np.mean(mat_dnn_test[:,2,3]), np.std(mat_dnn_test[:,2,3], ddof=1))
            }
    
    return results_dict





def compute_errors_and_rho_BO(Dt_gt, Fp_gt, Dp_gt, params_pred, bvals, nums):
    """
    Computes NRMSE, Spearmans rho, NMAE, parameter-RMSE, parameter-MAE, and true-singal-RMSE.
    """

    #norm
    normDt = np.mean(Dt_gt)
    normFp = np.mean(Fp_gt)
    normDp = np.mean(Dp_gt)
    
    # rmse
    rmse_Dt = np.sqrt(np.square(np.subtract(Dt_gt, params_pred[0])).mean())
    rmse_Fp = np.sqrt(np.square(np.subtract(Fp_gt, params_pred[1])).mean())
    rmse_Dp = np.sqrt(np.square(np.subtract(Dp_gt, params_pred[2])).mean())

    # mae
    mae_Dt = np.mean(np.abs(np.subtract(Dt_gt, params_pred[0])))
    mae_Fp = np.mean(np.abs(np.subtract(Fp_gt, params_pred[1])))
    mae_Dp = np.mean(np.abs(np.subtract(Dp_gt, params_pred[2])))
    
    # rho and p-value
    Spearman = np.zeros([3, 2])
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params_pred[0], params_pred[2])  # Dt-Dp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params_pred[0], params_pred[1])  # Dt-fp
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params_pred[1], params_pred[2])  # fp-Dp
    Spearman[np.isnan(Spearman)] = 1                    # if spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman = np.absolute(Spearman)                    # absolute value

    # parameter-rmse and parameter-mae
    params_pred_T = np.array([params_pred[0], params_pred[1], params_pred[2]]).T
    params_gt_T = np.array([Dt_gt, Fp_gt, Dp_gt]).T
    param_rmse = np.mean(np.sqrt(np.mean(np.square(np.subtract(params_gt_T, params_pred_T)), axis=-1)))
    param_mae = np.mean(np.abs(np.subtract(params_gt_T, params_pred_T)))

    # true-signal-rmse
    S0_gt = np.ones(nums)
    Sb_gt = from_utils.ivims(bvals, Dt_gt, Fp_gt, Dp_gt, S0_gt, nums)
    Sb_pred = from_utils.ivims(bvals, params_pred[0], params_pred[1], params_pred[2], params_pred[3], nums)
    mean_true_signal_rmse_pred = np.mean(from_utils.rmse(Sb_gt, Sb_pred))

    # norm, nrmse, rho, nmae
    mats = [[normDt, rmse_Dt/normDt, Spearman[0, 0], mae_Dt/normDt],
            [normFp, rmse_Fp/normFp, Spearman[1, 0], mae_Fp/normFp],
            [normDp, rmse_Dp/normDp, Spearman[2, 0], mae_Dp/normDp]]

    del params_pred
    return mats, param_rmse, param_mae, mean_true_signal_rmse_pred






def checkarg_simulation_params_BO(arg):
    if not hasattr(arg, 'bvalues'):
        warnings.warn('arg_sim.bvalues not defined. Using default value of [0, 50, 100, 800]')
        arg.bvalues = np.array([0, 50, 100, 800])
    if not hasattr(arg, 'num_samples_training'):
        warnings.warn('arg_sim.num_samples_training not defined. Using default of 100000')
        arg.num_samples_training = 100000
    if not hasattr(arg, 'num_samples_test'):
        warnings.warn('arg_sim.num_samples_test not defined. Using default of 100000')
        arg.num_samples_test = 100000
    if not hasattr(arg, 'repeats'):
        warnings.warn('arg_sim.repeats not defined. Using default value of 3')
        arg.repeats = 3
    if not hasattr(arg, 'rician'):
        warnings.warn('arg_sim.rician not defined. Using default of True')
        arg.rician = True
    if not hasattr(arg, 'ranges'):
        warnings.warn('arg_sim.ranges not defined. Using default of ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])')
        arg.ranges = ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])
    if not hasattr(arg, 'num_trials'):
        warnings.warn('arg_sim.num_trials not defined. Using default of 250')
        arg.num_trials = 250
    return arg