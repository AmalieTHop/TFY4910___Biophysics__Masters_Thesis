# import libraries
import bayesian_optimization.DNN_BO as from_DNN_BO
import algorithms.utils as from_utils    

import numpy as np
import time
import torch
import os
import warnings

import scipy.stats as scipy


def sim_dnn(arg_sim, snr, learning, arg_dnn, parameters, dir_out):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    #np.random.seed
    #numpy
    #torch.use_deterministic_algorithms(True)

    dims = 4

    # training
    signal_noisy_training, _, params_train_norm = sim_signal(snr, arg_sim.bvalues, 
                                                             num_samples=arg_sim.num_samples_training, 
                                                             ranges=arg_sim.ranges,
                                                             rician=arg_sim.rician, 
                                                             state=16)
    
    # validation
    signal_noisy_valHT, params_valHT_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=4)

    # test
    signal_noisy_test, params_test_unorm, _ = sim_signal(snr, arg_sim.bvalues, 
                                                      num_samples=arg_sim.num_samples_test, 
                                                      ranges=arg_sim.ranges,
                                                      rician=arg_sim.rician, 
                                                      state=18)


    
    # prepare a larger array in case we repeat training
    params_dnn_valHT = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])
    params_dnn_test = np.zeros([arg_sim.repeats, dims+1, arg_sim.num_samples_test])

    losses_valES = np.zeros([arg_sim.repeats])
    losses_valHT = np.zeros([arg_sim.repeats])
    losses_test = np.zeros([arg_sim.repeats])


    # loop over repeats
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

        # save trained network
        #torch.save(net, os.path.join(dir_out, f'trained_model_net_{aa}'))
        #np.save(os.path.join(dir_out, f'loss_train_{aa}'), loss_train)
        #np.save(os.path.join(dir_out, f'loss_valES_{aa}'), loss_valES)
        #np.save(os.path.join(dir_out, f'loss_valES_best_{aa}'), losses_valES[aa])
        #print('Trained model is saved')
        

        #net = torch.load(f'/Volumes/LaCie/TFY4910/dataA/output/test_projectsettingsslf_mae_relu6/EMIN_1064/dnn/trained_model_net')#torch.load(f'../dataA/simulations_test/rep5_linear/SNR{SNR}/trained_model_net_0')  

        # predict parameters on validation_HT set
        start_time = time.time()
        params_dnn_valHT[aa], losses_valHT[aa] = from_DNN_BO.predict_IVIM(signal_noisy_valHT, arg_sim.bvalues, net, arg_dnn, parameters)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        #np.save(os.path.join(dir_out, f'loss_valHT_{aa}'), losses_valHT[aa])


        # predict parameters on test set
        start_time = time.time()
        params_dnn_test[aa], losses_test[aa] = from_DNN_BO.predict_IVIM(signal_noisy_test, arg_sim.bvalues, net, arg_dnn, parameters)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        #np.save(os.path.join(dir_out, f'loss_test_{aa}'), losses_test[aa])


        # remove network to save memory
        del net
        if arg_dnn.train_pars.use_cuda:
            torch.cuda.empty_cache()
            

    print('Results for dnn')

    mat_dnn_valHT = np.zeros([arg_sim.repeats, dims-1, 4])
    mat_dnn_test = np.zeros([arg_sim.repeats, dims-1, 4])

    param_rmse_valHT = np.zeros([arg_sim.repeats])
    param_rmse_test = np.zeros([arg_sim.repeats])

    param_mae_valHT = np.zeros([arg_sim.repeats])
    param_mae_test = np.zeros([arg_sim.repeats])

    mean_true_signal_rmse_pred_valHT = np.zeros([arg_sim.repeats])
    mean_true_signal_rmse_pred_test = np.zeros([arg_sim.repeats])

    # determine errors and Spearman Rank
    for aa in range(arg_sim.repeats):
        mat_dnn_valHT[aa], param_rmse_valHT[aa], param_mae_valHT[aa], mean_true_signal_rmse_pred_valHT[aa] = print_errors(params_valHT_unorm[0], params_valHT_unorm[1], params_valHT_unorm[2], params_dnn_valHT[aa], arg_sim.bvalues, arg_sim.num_samples_test)
        mat_dnn_test[aa], param_rmse_test[aa], param_mae_test[aa], mean_true_signal_rmse_pred_test[aa] = print_errors(params_test_unorm[0], params_test_unorm[1], params_test_unorm[2], params_dnn_test[aa], arg_sim.bvalues, arg_sim.num_samples_test)
    #np.save(os.path.join(dir_out, f'mat_dnn_valHT'), mat_dnn_valHT)
    #np.save(os.path.join(dir_out, f'param_rmse_valHT'), param_rmse_valHT)
    #np.save(os.path.join(dir_out, f'param_mae_valHT'), param_mae_valHT)
    #np.save(os.path.join(dir_out, f'mean_true_signal_rmse_pred_valHT'), mean_true_signal_rmse_pred_valHT)
    #np.save(os.path.join(dir_out, f'mat_dnn_test'), mat_dnn_test)
    #np.save(os.path.join(dir_out, f'param_rmse_test'), param_rmse_test)
    #np.save(os.path.join(dir_out, f'param_mae_test'), param_mae_test)
    #np.save(os.path.join(dir_out, f'mean_true_signal_rmse_pred_test'), mean_true_signal_rmse_pred_test)

    # calculate Stability Factor
    if arg_sim.repeats > 1:
        stability_valHT = np.sqrt(np.mean(np.square(np.std(params_dnn_valHT, axis=0)), axis=1))
        stability_test = np.sqrt(np.mean(np.square(np.std(params_dnn_valHT, axis=0)), axis=1))
        stability_valHT = stability_valHT[[0, 1, 2]] / [np.mean(params_valHT_unorm[0]), np.mean(params_valHT_unorm[1]), np.mean(params_valHT_unorm[2])]
        stability_test = stability_test[[0, 1, 2]] / [np.mean(params_test_unorm[0]), np.mean(params_test_unorm[1]), np.mean(params_test_unorm[2])]
    else:
        stability_valHT = np.zeros(dims-1)
        stability_test = np.zeros(dims-1)
        
    # save
    #np.save(os.path.join(dir_out, f'params_dnn_valHT'), params_dnn_valHT)
    #np.save(os.path.join(dir_out, f'params_dnn_test'), params_dnn_test)
    #np.save(os.path.join(dir_out, f'stability_valHT'), stability_valHT)
    #np.save(os.path.join(dir_out, f'stability_test'), stability_test)


    del params_dnn_valHT
    del params_dnn_test


    sums_nrsme_valHT = mat_dnn_valHT[:,0,1] + mat_dnn_valHT[:,1,1] + mat_dnn_valHT[:,2,1]
    mean_sum_nrsme_valHT = np.mean(sums_nrsme_valHT)
    std_sum_nrmse_valHT = np.std(sums_nrsme_valHT, ddof=1)

    sums_nrsme_test = mat_dnn_test[:,0,1] + mat_dnn_test[:,1,1] + mat_dnn_test[:,2,1]
    mean_sum_nrsme_test = np.mean(sums_nrsme_test)
    std_sum_nrmse_test = np.std(sums_nrsme_test, ddof=1)

    sums_nmae_valHT = mat_dnn_valHT[:,0,3] + mat_dnn_valHT[:,1,3] + mat_dnn_valHT[:,2,3]
    mean_sum_nmae_valHT = np.mean(sums_nmae_valHT)
    std_sum_nmae_valHT = np.std(sums_nmae_valHT, ddof=1)

    sums_nmae_test = mat_dnn_test[:,0,3] + mat_dnn_test[:,1,3] + mat_dnn_test[:,2,3]
    mean_sum_nmae_test = np.mean(sums_nmae_test)
    std_sum_nmae_test = np.std(sums_nmae_test, ddof=1)


    return {"loss_valHT": (np.mean(losses_valHT), np.std(losses_valHT, ddof=1)), 
            "loss_valES": (np.mean(losses_valES), np.std(losses_valES, ddof=1)), 
            "loss_test": (np.mean(losses_test), np.std(losses_test, ddof=1)), 

            "sum_nrmse_valHT": (mean_sum_nrsme_valHT, std_sum_nrmse_valHT),
            "sum_nrmse_test": (mean_sum_nrsme_test, std_sum_nrmse_test),

            "sum_nmae_valHT": (mean_sum_nmae_valHT, std_sum_nmae_valHT),
            "sum_nmae_test": (mean_sum_nmae_test, std_sum_nmae_test),

            "param_rmse_valHT": (np.mean(param_rmse_valHT), np.std(param_rmse_valHT, ddof=1)),
            "param_rmse_test": (np.mean(param_rmse_test), np.std(param_rmse_test, ddof=1)),

            "param_mae_valHT": (np.mean(param_mae_valHT), np.std(param_mae_valHT, ddof=1)),
            "param_mae_test": (np.mean(param_mae_test), np.std(param_mae_test, ddof=1)),

            "true_signal_rmse_valHT": (np.mean(mean_true_signal_rmse_pred_valHT), np.std(mean_true_signal_rmse_pred_valHT, ddof=1)),
            "true_signal_rmse_test": (np.mean(mean_true_signal_rmse_pred_test), np.std(mean_true_signal_rmse_pred_test, ddof=1)),

            "Dt_nrmse_valHT": (np.mean(mat_dnn_valHT[:,0,1]), np.std(mat_dnn_valHT[:,0,1], ddof=1)), 
            "Fp_nrmse_valHT": (np.mean(mat_dnn_valHT[:,1,1]), np.std(mat_dnn_valHT[:,1,1], ddof=1)),
            "Dp_nrmse_valHT": (np.mean(mat_dnn_valHT[:,2,1]), np.std(mat_dnn_valHT[:,2,1], ddof=1)),

            "Dt_nrmse_test": (np.mean(mat_dnn_test[:,0,1]), np.std(mat_dnn_test[:,0,1], ddof=1)), 
            "Fp_nrmse_test": (np.mean(mat_dnn_test[:,1,1]), np.std(mat_dnn_test[:,1,1], ddof=1)),
            "Dp_nrmse_test": (np.mean(mat_dnn_test[:,2,1]), np.std(mat_dnn_test[:,2,1], ddof=1)),
            
            "Dt_nmae_valHT": (np.mean(mat_dnn_valHT[:,0,3]), np.std(mat_dnn_valHT[:,0,3], ddof=1)), 
            "Fp_nmae_valHT": (np.mean(mat_dnn_valHT[:,1,3]), np.std(mat_dnn_valHT[:,1,3], ddof=1)),
            "Dp_nmae_valHT": (np.mean(mat_dnn_valHT[:,2,3]), np.std(mat_dnn_valHT[:,2,3], ddof=1)),

            "Dt_nmae_test": (np.mean(mat_dnn_test[:,0,3]), np.std(mat_dnn_test[:,0,3], ddof=1)), 
            "Fp_nmae_test": (np.mean(mat_dnn_test[:,1,3]), np.std(mat_dnn_test[:,1,3], ddof=1)),
            "Dp_nmae_test": (np.mean(mat_dnn_test[:,2,3]), np.std(mat_dnn_test[:,2,3], ddof=1))
            }





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



def print_errors(Dt_gt, Fp_gt, Dp_gt, params_pred, bvals, nums):
    S0_gt = np.ones(nums)

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
    

    # rho
    Spearman = np.zeros([3, 2])
    Spearman[0, 0], Spearman[0, 1] = scipy.spearmanr(params_pred[0], params_pred[2])  # DvDp
    Spearman[1, 0], Spearman[1, 1] = scipy.spearmanr(params_pred[0], params_pred[1])  # Dvf
    Spearman[2, 0], Spearman[2, 1] = scipy.spearmanr(params_pred[1], params_pred[2])  # fvDp
    Spearman[np.isnan(Spearman)] = 1                                        # if spearman is nan, set as 1 (because of constant estimated IVIM parameters)
    Spearman = np.absolute(Spearman)                                        # absolute Spearman

    
    params_pred_T = np.array([params_pred[0], params_pred[1], params_pred[2]]).T
    params_gt_T = np.array([Dt_gt, Fp_gt, Dp_gt]).T
    param_rmse = np.mean(np.sqrt(np.mean(np.square(np.subtract(params_gt_T, params_pred_T)), axis=-1)))
    param_mae = np.mean(np.abs(np.subtract(params_gt_T, params_pred_T)))


    Sb_gt = from_utils.ivims(bvals, Dt_gt, Fp_gt, Dp_gt, S0_gt, nums)
    Sb_pred = from_utils.ivims(bvals, params_pred[0], params_pred[1], params_pred[2], params_pred[3], nums)
    mean_true_signal_rmse_pred = np.mean(from_utils.rmse(Sb_gt, Sb_pred))


    # norm, nrmse, rho, nmae, param_rmse, param_mae, true_signal_rmse
    mats = [[normDt, rmse_Dt/normDt, Spearman[0, 0], mae_Dt/normDt],
            [normFp, rmse_Fp/normFp, Spearman[1, 0], mae_Fp/normFp],
            [normDp, rmse_Dp/normDp, Spearman[2, 0], mae_Dp/normDp]]
    print(f'Results from NN: {mats}')

    del params_pred
    return mats, param_rmse, param_mae, mean_true_signal_rmse_pred






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
        warnings.warn('arg_sim.num_samples_training not defined. Using default of 10000000')
        arg.num_samples_training = 100000
    if not hasattr(arg, 'num_samples_test'):
        warnings.warn('arg_sim.num_samples_test not defined. Using default of 100000')
        arg.num_samples_test = 100000
    if not hasattr(arg, 'ranges'):
        warnings.warn('arg_sim.ranges not defined. Using default of ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])')
        arg.ranges = ([0.0005, 0.05, 0.005], [0.003, 0.50, 0.1])
    return arg