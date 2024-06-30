import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()


import os
import copy

import analysis.utils as from_utils

from tueplots import figsizes, fonts


class Simplt_bvalset:
    def __init__(self, snrs, bvals, reps, exp_name, num_load=100):
        self.snrs = [str(x) for x in snrs]
        self.bvals = bvals
        self.reps = reps
        self.exp_name = exp_name
        self.num_load = num_load
    

        self.sim_bounds = np.array([[0.0005, 0.05, 0.005], [0.003, 0.50, 0.1]])
        self.max = 800
        self.bval_range_plt = np.linspace(0, self.max, self.max+1)

        self.dir_out = f'../simulations/simulations_plot/b{len(self.bvals)}/{exp_name}'
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)


        self.prepare_dfs() ###



    def prepare_dfs(self):
        if len(self.bvals) == 11:
            idx_4b = [True, True, True, True, True, True, True, True, True, True, True]
        elif len(self.bvals) == 5:
            idx_4b = [True, True, True, True, True]
        elif len(self.bvals) == 4:
            idx_4b = [True, True, True, True]


        # load ground truth
        self.Dt_gt = np.load('../simulations/simulations_data/Dt_gt.npy')[:self.num_load]
        self.Fp_gt = np.load('../simulations/simulations_data/Fp_gt.npy')[:self.num_load]
        self.Dp_gt = np.load('../simulations/simulations_data/Dp_gt.npy')[:self.num_load]
        self.S0_gt = np.ones(self.num_load)
        Sb_gt = from_utils.ivims(self.bvals, self.Dt_gt, self.Fp_gt, self.Dp_gt, self.S0_gt, self.num_load)[:, idx_4b]

        # load estimated data
        zero_arr = np.zeros((len(self.snrs), self.num_load))
        self.measured_signal_rmse_gt = copy.deepcopy(zero_arr)
        self.Dt_lsq, self.Fp_lsq, self.Dp_lsq, self.S0_lsq, self.rmse_lsq = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        self.Dt_seg, self.Fp_seg, self.Dp_seg, self.S0_seg, self.rmse_seg = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        zero_arr = np.zeros((len(self.snrs), self.reps, self.num_load))
        self.Dt_dnn_sup, self.Fp_dnn_sup, self.Dp_dnn_sup, self.S0_dnn_sup, self.rmse_dnn_sup = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)
        self.Dt_dnn_slf, self.Fp_dnn_slf, self.Dp_dnn_slf, self.S0_dnn_slf, self.rmse_dnn_slf = copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr), copy.deepcopy(zero_arr)

        for i, snr in enumerate(self.snrs):
            self.measured_signal_rmse_gt[i] = np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/measured_signal_rmse_test_gt.npy')[:self.num_load]
            self.Dt_lsq[i], self.Fp_lsq[i], self.Dp_lsq[i], self.S0_lsq[i], self.rmse_lsq[i] = np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/lsq/params_lsq.npy')[:, :self.num_load]
            self.Dt_seg[i], self.Fp_seg[i], self.Dp_seg[i], self.S0_seg[i], self.rmse_seg[i] = np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/seg/params_seg.npy')[:, :self.num_load]

            self.Dt_dnn_sup[i], self.Fp_dnn_sup[i], self.Dp_dnn_sup[i], self.S0_dnn_sup[i], self.rmse_dnn_sup[i] = np.swapaxes(np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/dnn_sup/{self.exp_name}/params_dnn.npy')[:, : , :self.num_load], 0, 1)
            self.Dt_dnn_slf[i], self.Fp_dnn_slf[i], self.Dp_dnn_slf[i], self.S0_dnn_slf[i], self.rmse_dnn_slf[i] = np.swapaxes(np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/dnn_slf/{self.exp_name}/params_dnn.npy')[:, : , :self.num_load], 0, 1)

   


        # prepare dataframes
        df_nrmse = pd.DataFrame(columns=['SNR', 'param', 'method', 'param_val'])
        df_rho = pd.DataFrame(columns=['SNR', 'p', 'method', 'p_val'])
        df_cv = pd.DataFrame(columns=['SNR', 'param', 'method', 'cv_val'])
        df_measured_signal_rmse = pd.DataFrame(columns=['SNR', 'method', 'measured_signal_rmse_val'])
        df_true_signal_rmse = pd.DataFrame(columns=['SNR', 'method', 'true_signal_rmse_val'])

        for snr_idx, snr in enumerate(self.snrs):
            dir_in_snr = f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}'

            # load measured signal
            Sb_noisy = np.load(f'../simulations/simulations_data/b{len(self.bvals)}/snr{snr}/signal_noisy_test.npy')[:self.num_load, idx_4b]

            # nmae
            mat_lsq = np.load(os.path.join(dir_in_snr, f'lsq/mat_lsq.npy'))
            mat_seg = np.load(os.path.join(dir_in_snr, f'seg/mat_seg.npy'))
            mat_dnn_sup = np.load(os.path.join(dir_in_snr, f'dnn_sup/{self.exp_name}/mat_dnn_raw.npy'))
            mat_dnn_slf = np.load(os.path.join(dir_in_snr, f'dnn_slf/{self.exp_name}/mat_dnn_raw.npy'))

            # stability
            stability_dnn_sup = np.load(os.path.join(dir_in_snr, f'dnn_sup/{self.exp_name}/stability.npy'))
            stability_dnn_slf = np.load(os.path.join(dir_in_snr, f'dnn_slf/{self.exp_name}/stability.npy'))

            # true signal rmse
            Sb_lsq = from_utils.ivims(self.bvals, self.Dt_lsq[snr_idx], self.Fp_lsq[snr_idx], self.Dp_lsq[snr_idx], self.S0_lsq[snr_idx], self.num_load)[:, idx_4b]
            Sb_seg = from_utils.ivims(self.bvals, self.Dt_seg[snr_idx], self.Fp_seg[snr_idx], self.Dp_seg[snr_idx], self.S0_seg[snr_idx], self.num_load)[:, idx_4b]
            mean_true_signal_rmse_lsq = np.mean(from_utils.rmse(Sb_gt, Sb_lsq))
            mean_true_signal_rmse_seg = np.mean(from_utils.rmse(Sb_gt, Sb_seg))

            self.rmse_lsq[snr_idx] = from_utils.rmse(Sb_noisy, Sb_lsq)
            self.rmse_seg[snr_idx] = from_utils.rmse(Sb_noisy, Sb_seg)

            self.measured_signal_rmse_gt[snr_idx] = from_utils.rmse(Sb_noisy, Sb_gt)
            measured_signal_rmse_gt = pd.DataFrame({'SNR': snr, 'method': 'GT', 'measured_signal_rmse_val': [np.mean(self.measured_signal_rmse_gt[snr_idx])]})

            nrmse_lsq_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': f'LSQ', 'param_val': [mat_lsq[0,3]]})
            nrmse_lsq_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': f'LSQ', 'param_val': [mat_lsq[1,3]]})
            nrmse_lsq_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': f'LSQ', 'param_val': [mat_lsq[2,3]]})
            sp_lsq_Dt = pd.DataFrame({'SNR': snr, 'p': 'DtDp', 'method': f'LSQ', 'p_val': [mat_lsq[0,2]]})
            sp_lsq_Fp = pd.DataFrame({'SNR': snr, 'p': 'DtFp', 'method': f'LSQ', 'p_val': [mat_lsq[1,2]]})
            sp_lsq_Dp = pd.DataFrame({'SNR': snr, 'p': 'DpFp', 'method': f'LSQ', 'p_val': [mat_lsq[2,2]]})
            measured_signal_rmse_lsq = pd.DataFrame({'SNR': snr, 'method': f'LSQ', 'measured_signal_rmse_val': [np.mean(self.rmse_lsq[snr_idx])]})
            true_signal_rmse_lsq = pd.DataFrame({'SNR': snr, 'method': f'LSQ', 'true_signal_rmse_val': [mean_true_signal_rmse_lsq]})

            nrmse_seg_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': f'SEG', 'param_val': [mat_seg[0,3]]})
            nrmse_seg_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': f'SEG', 'param_val': [mat_seg[1,3]]})
            nrmse_seg_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': f'SEG', 'param_val': [mat_seg[2,3]]})
            sp_seg_Dt = pd.DataFrame({'SNR': snr, 'p': 'DtDp', 'method': f'SEG', 'p_val': [mat_seg[0,2]]})
            sp_seg_Fp = pd.DataFrame({'SNR': snr, 'p': 'DtFp', 'method': f'SEG', 'p_val': [mat_seg[1,2]]})
            sp_seg_Dp = pd.DataFrame({'SNR': snr, 'p': 'DpFp', 'method': f'SEG', 'p_val': [mat_seg[2,2]]})
            measured_signal_rmse_seg = pd.DataFrame({'SNR': snr, 'method': f'SEG', 'measured_signal_rmse_val': [np.mean(self.rmse_seg[snr_idx])]})
            true_signal_rmse_seg = pd.DataFrame({'SNR': snr, 'method': f'SEG', 'true_signal_rmse_val': [mean_true_signal_rmse_seg]})

            df_nrmse = pd.concat([df_nrmse, nrmse_lsq_Dt, nrmse_lsq_Fp, nrmse_lsq_Dp, 
                                            nrmse_seg_Dt, nrmse_seg_Fp, nrmse_seg_Dp
                                            ])
            df_rho = pd.concat([df_rho, sp_lsq_Dt, sp_lsq_Fp, sp_lsq_Dp, 
                                        sp_seg_Dt, sp_seg_Fp, sp_seg_Dp
                                        ]) 
            df_measured_signal_rmse = pd.concat([df_measured_signal_rmse, measured_signal_rmse_gt, measured_signal_rmse_lsq, measured_signal_rmse_seg])
            df_true_signal_rmse = pd.concat([df_true_signal_rmse, true_signal_rmse_lsq, true_signal_rmse_seg])


            for i in range(self.reps):
                Sb_dnn_sup = from_utils.ivims(self.bvals, self.Dt_dnn_sup[snr_idx][i], self.Fp_dnn_sup[snr_idx][i], self.Dp_dnn_sup[snr_idx][i], self.S0_dnn_sup[snr_idx][i], self.num_load)[:, idx_4b]
                Sb_dnn_slf = from_utils.ivims(self.bvals, self.Dt_dnn_slf[snr_idx][i], self.Fp_dnn_slf[snr_idx][i], self.Dp_dnn_slf[snr_idx][i], self.S0_dnn_slf[snr_idx][i], self.num_load)[:, idx_4b]
                mean_true_signal_rmse_dnn_sup = np.mean(from_utils.rmse(Sb_gt, Sb_dnn_sup))
                mean_true_signal_rmse_dnn_slf = np.mean(from_utils.rmse(Sb_gt, Sb_dnn_slf))

                self.rmse_dnn_sup[snr_idx, i] = from_utils.rmse(Sb_noisy, Sb_dnn_sup)
                self.rmse_dnn_slf[snr_idx, i] = from_utils.rmse(Sb_noisy, Sb_dnn_slf)


                nrmse_dnn_sup_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': r'$\mathregular{DNN_{SL}}$', 'param_val': [mat_dnn_sup[i,0,3]]})
                nrmse_dnn_sup_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': r'$\mathregular{DNN_{SL}}$', 'param_val': [mat_dnn_sup[i,1,3]]})
                nrmse_dnn_sup_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': r'$\mathregular{DNN_{SL}}$', 'param_val': [mat_dnn_sup[i,2,3]]})
                sp_dnn_sup_Dt = pd.DataFrame({'SNR': snr, 'p': 'DtDp', 'method': r'$\mathregular{DNN_{SL}}$', 'p_val': [mat_dnn_sup[i,0,2]]})
                sp_dnn_sup_Fp = pd.DataFrame({'SNR': snr, 'p': 'DtFp', 'method': r'$\mathregular{DNN_{SL}}$', 'p_val': [mat_dnn_sup[i,1,2]]})
                sp_dnn_sup_Dp = pd.DataFrame({'SNR': snr, 'p': 'DpFp', 'method': r'$\mathregular{DNN_{SL}}$', 'p_val': [mat_dnn_sup[i,2,2]]})
                measured_signal_rmse_dnn_sup = pd.DataFrame({'SNR': snr, 'method': r'$\mathregular{DNN_{SL}}$', 'measured_signal_rmse_val': [np.mean(self.rmse_dnn_sup[snr_idx, i])]})
                true_signal_rmse_dnn_sup = pd.DataFrame({'SNR': snr, 'method': r'$\mathregular{DNN_{SL}}$', 'true_signal_rmse_val': [mean_true_signal_rmse_dnn_sup]})

                nrmse_dnn_slf_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': r'$\mathregular{DNN_{SSL}}$', 'param_val': [mat_dnn_slf[i,0,3]]})
                nrmse_dnn_slf_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': r'$\mathregular{DNN_{SSL}}$', 'param_val': [mat_dnn_slf[i,1,3]]})
                nrmse_dnn_slf_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': r'$\mathregular{DNN_{SSL}}$', 'param_val': [mat_dnn_slf[i,2,3]]})
                sp_dnn_slf_Dt = pd.DataFrame({'SNR': snr, 'p': 'DtDp', 'method': r'$\mathregular{DNN_{SSL}}$', 'p_val': [mat_dnn_slf[i,0,2]]})
                sp_dnn_slf_Fp = pd.DataFrame({'SNR': snr, 'p': 'DtFp', 'method': r'$\mathregular{DNN_{SSL}}$', 'p_val': [mat_dnn_slf[i,1,2]]})
                sp_dnn_slf_Dp = pd.DataFrame({'SNR': snr, 'p': 'DpFp', 'method': r'$\mathregular{DNN_{SSL}}$', 'p_val': [mat_dnn_slf[i,2,2]]})
                measured_signal_rmse_dnn_slf = pd.DataFrame({'SNR': snr, 'method': r'$\mathregular{DNN_{SSL}}$', 'measured_signal_rmse_val': [np.mean(self.rmse_dnn_slf[snr_idx, i])]})
                true_signal_rmse_dnn_slf = pd.DataFrame({'SNR': snr, 'method': r'$\mathregular{DNN_{SSL}}$', 'true_signal_rmse_val': [mean_true_signal_rmse_dnn_slf]})


                df_nrmse = pd.concat([df_nrmse,
                                      nrmse_dnn_sup_Dt, nrmse_dnn_sup_Fp, nrmse_dnn_sup_Dp,
                                      nrmse_dnn_slf_Dt, nrmse_dnn_slf_Fp, nrmse_dnn_slf_Dp
                                      ]) 
                df_rho = pd.concat([df_rho, 
                                    sp_dnn_sup_Dt, sp_dnn_sup_Fp, sp_dnn_sup_Dp,
                                    sp_dnn_slf_Dt, sp_dnn_slf_Fp, sp_dnn_slf_Dp
                                    ])
                df_measured_signal_rmse = pd.concat([df_measured_signal_rmse, measured_signal_rmse_dnn_sup, measured_signal_rmse_dnn_slf])
                df_true_signal_rmse = pd.concat([df_true_signal_rmse, true_signal_rmse_dnn_sup, true_signal_rmse_dnn_slf])

            cv_dnn_sup_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': r'$\mathregular{DNN_{SL}}$', 'cv_val': [stability_dnn_sup[0]]})
            cv_dnn_sup_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': r'$\mathregular{DNN_{SL}}$', 'cv_val': [stability_dnn_sup[1]]})
            cv_dnn_sup_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': r'$\mathregular{DNN_{SL}}$', 'cv_val': [stability_dnn_sup[2]]})

            cv_dnn_slf_Dt = pd.DataFrame({'SNR': snr, 'param': 'Dt', 'method': r'$\mathregular{DNN_{SSL}}$', 'cv_val': [stability_dnn_slf[0]]})
            cv_dnn_slf_Fp = pd.DataFrame({'SNR': snr, 'param': 'Fp', 'method': r'$\mathregular{DNN_{SSL}}$', 'cv_val': [stability_dnn_slf[1]]})
            cv_dnn_slf_Dp = pd.DataFrame({'SNR': snr, 'param': 'Dp', 'method': r'$\mathregular{DNN_{SSL}}$', 'cv_val': [stability_dnn_slf[2]]})

            df_cv = pd.concat([df_cv, 
                               cv_dnn_sup_Dt, cv_dnn_sup_Fp, cv_dnn_sup_Dp,
                               cv_dnn_slf_Dt, cv_dnn_slf_Fp, cv_dnn_slf_Dp,
                               ])

        df_nrmse.to_csv(self.dir_out + '/df_nrmse.csv', index=False)
        df_rho.to_csv(self.dir_out + '/df_rho.csv', index=False)
        df_cv.to_csv(self.dir_out + '/df_cv.csv', index=False)
        df_measured_signal_rmse.to_csv(self.dir_out + '/df_measured_signal_rmse.csv', index=False)
        df_true_signal_rmse.to_csv(self.dir_out + '/df_true_signal_rmse.csv', index=False)


    def plot_param_space(self):
        
        sns.set_theme()
        params = {'axes.labelsize': 30,
                  'axes.grid' : True, 
                  'axes.titlesize': 30,
                  'lines.marker': 'none',
                  'lines.markersize': 1, #0.1,
                  'scatter.marker': 'o',
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)
        algo_bounds = np.array([[0, 0, 0.005], [0.005, 0.7, 0.3]])
         
        for snr_idx, snr in enumerate(self.snrs):
            
            fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize=(20, 20))
            [(ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23)] = axes


            scatter00 = ax00.scatter(x=self.Dp_lsq[snr_idx], y=self.Dt_lsq[snr_idx])
            scatter01 = ax01.scatter(x=self.Dp_seg[snr_idx], y=self.Dt_seg[snr_idx])
            scatter02 = ax02.scatter(x=self.Dp_dnn_sup[0][snr_idx], y=self.Dt_dnn_sup[0][snr_idx],)
            scatter03 = ax03.scatter(x=self.Dp_dnn_slf[0][snr_idx], y=self.Dt_dnn_slf[0][snr_idx])

            scatter10 = ax10.scatter(x=self.Dt_lsq[snr_idx], y=self.Fp_lsq[snr_idx])
            scatter11 = ax11.scatter(x=self.Dt_seg[snr_idx], y=self.Fp_seg[snr_idx])
            scatter12 = ax12.scatter(x=self.Dt_dnn_sup[0][snr_idx], y=self.Fp_dnn_sup[0][snr_idx])
            scatter13 = ax13.scatter(x=self.Dt_dnn_slf[0][snr_idx], y=self.Fp_dnn_slf[0][snr_idx])

            scatter20 = ax20.scatter(x=self.Fp_lsq[snr_idx], y=self.Dp_lsq[snr_idx])
            scatter21 = ax21.scatter(x=self.Fp_seg[snr_idx], y=self.Dp_seg[snr_idx])
            scatter22 = ax22.scatter(x=self.Fp_dnn_sup[0][snr_idx], y=self.Dp_dnn_sup[0][snr_idx])
            scatter23 = ax23.scatter(x=self.Fp_dnn_slf[0][snr_idx], y=self.Dp_dnn_slf[0][snr_idx])



            # Dummies
            scatterd00 = ax00.scatter(x=[algo_bounds[0, 2], algo_bounds[1, 2]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd10 = ax01.scatter(x=[algo_bounds[0, 2], algo_bounds[1, 2]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd20 = ax02.scatter(x=[algo_bounds[0, 2], algo_bounds[1, 2]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd20 = ax03.scatter(x=[algo_bounds[0, 2], algo_bounds[1, 2]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)

            scatterd01 = ax10.scatter(x=[algo_bounds[0, 0], algo_bounds[1, 0]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd11 = ax11.scatter(x=[algo_bounds[0, 0], algo_bounds[1, 0]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd21 = ax12.scatter(x=[algo_bounds[0, 0], algo_bounds[1, 0]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd21 = ax13.scatter(x=[algo_bounds[0, 0], algo_bounds[1, 0]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)

            scatterd02 = ax20.scatter(x=[algo_bounds[0, 1], algo_bounds[1, 1]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)
            scatterd12 = ax21.scatter(x=[algo_bounds[0, 1], algo_bounds[1, 1]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)
            scatterd22 = ax22.scatter(x=[algo_bounds[0, 1], algo_bounds[1, 1]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)
            scatterd22 = ax23.scatter(x=[algo_bounds[0, 1], algo_bounds[1, 1]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)



            column_titles = ['LSQ', 'SEG', r'$\mathregular{DNN_{SL}}$', r'$\mathregular{DNN_{SSL}}$']
            pad = 50
            for ax, column_title in zip(axes[0], column_titles):
                ax.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', fontsize=40)









    def plot_pred_vs_truth(self, colordim=''):

        ref_Dt = np.linspace(self.sim_bounds[0, 0], self.sim_bounds[1, 0], 100)
        ref_Fp = np.linspace(self.sim_bounds[0, 1], self.sim_bounds[1, 1], 100)
        ref_Dp = np.linspace(self.sim_bounds[0, 2], self.sim_bounds[1, 2], 100)


        sns.set_theme()
        params = {'axes.labelsize': 30,
                  'axes.grid' : True, 
                  'axes.titlesize': 30,
                  'lines.marker': 'none',
                  'lines.markersize': 0.1,
                  'scatter.marker': '.',
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)
        algo_bounds = np.array([[0, 0, 0.005], [0.005, 0.7, 0.3]])


        for snr_idx, snr in enumerate(self.snrs):
            if colordim == 'Dt':
                c = [self.Dt_gt, self.Dt_gt, self.Dt_gt, self.Dt_gt]
                c_label = r'$D_t$ [mm$^2$/s]'
            elif colordim == 'Fp':
                c = [self.Fp_gt, self.Fp_gt, self.Fp_gt, self.Fp_gt]
                c_label = r'$f_p$'
            elif colordim == 'Dp':
                c = [self.Dp_gt, self.Dp_gt, self.Dp_gt, self.Dp_gt]
                c_label = r'$D_p$ [mm$^2$/s]'
            elif colordim == 'rmse':
                c = [self.rmse_lsq[snr_idx], self.rmse_seg[snr_idx], np.mean(self.rmse_dnn_sup[snr_idx], axis=0), np.mean(self.rmse_dnn_slf[snr_idx], axis=0)]
                c_label = r'signal-RMSE'
            else:
                c = ['steelblue', 'coral', 'palevioletred', 'mediumseagreen']
                c_label = None

            fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize=(25, 20))
            [(ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23)] = axes


            scatter00 = ax00.scatter(x=self.Dt_gt, y=self.Dt_lsq[snr_idx], c=c[0])
            scatter10 = ax10.scatter(x=self.Fp_gt, y=self.Fp_lsq[snr_idx], c=c[0])
            scatter20 = ax20.scatter(x=self.Dp_gt, y=self.Dp_lsq[snr_idx], c=c[0])

            scatter01 = ax01.scatter(x=self.Dt_gt, y=self.Dt_seg[snr_idx], c=c[1])
            scatter11 = ax11.scatter(x=self.Fp_gt, y=self.Fp_seg[snr_idx], c=c[1])
            scatter21 = ax21.scatter(x=self.Dp_gt, y=self.Dp_seg[snr_idx], c=c[1])

            scatter02 = ax02.scatter(x=self.Dt_gt, y=np.mean(self.Dt_dnn_sup[snr_idx], axis=0), c=c[2])
            scatter12 = ax12.scatter(x=self.Fp_gt, y=np.mean(self.Fp_dnn_sup[snr_idx], axis=0), c=c[2])
            scatter22 = ax22.scatter(x=self.Dp_gt, y=np.mean(self.Dp_dnn_sup[snr_idx], axis=0), c=c[2])

            scatter03 = ax03.scatter(x=self.Dt_gt, y=np.mean(self.Dt_dnn_slf[snr_idx], axis=0), c=c[3])
            scatter13 = ax13.scatter(x=self.Fp_gt, y=np.mean(self.Fp_dnn_slf[snr_idx], axis=0), c=c[3])
            scatter23 = ax23.scatter(x=self.Dp_gt, y=np.mean(self.Dp_dnn_slf[snr_idx], axis=0), c=c[3])


            # dummies
            scatterd00 = ax00.scatter(x=[self.sim_bounds[0, 0], self.sim_bounds[1, 0]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd10 = ax10.scatter(x=[self.sim_bounds[0, 1], self.sim_bounds[1, 1]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd20 = ax20.scatter(x=[self.sim_bounds[0, 2], self.sim_bounds[1, 2]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)
            
            scatterd01 = ax01.scatter(x=[self.sim_bounds[0, 0], self.sim_bounds[1, 0]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd11 = ax11.scatter(x=[self.sim_bounds[0, 1], self.sim_bounds[1, 1]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd21 = ax21.scatter(x=[self.sim_bounds[0, 2], self.sim_bounds[1, 2]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)

            scatterd02 = ax02.scatter(x=[self.sim_bounds[0, 0], self.sim_bounds[1, 0]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd12 = ax12.scatter(x=[self.sim_bounds[0, 1], self.sim_bounds[1, 1]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd22 = ax22.scatter(x=[self.sim_bounds[0, 2], self.sim_bounds[1, 2]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)

            scatterd03 = ax03.scatter(x=[self.sim_bounds[0, 0], self.sim_bounds[1, 0]], y=[algo_bounds[0, 0], algo_bounds[1, 0]], alpha=0)
            scatterd13 = ax13.scatter(x=[self.sim_bounds[0, 1], self.sim_bounds[1, 1]], y=[algo_bounds[0, 1], algo_bounds[1, 1]], alpha=0)
            scatterd23 = ax23.scatter(x=[self.sim_bounds[0, 2], self.sim_bounds[1, 2]], y=[algo_bounds[0, 2], algo_bounds[1, 2]], alpha=0)


            if colordim:
                cbar00 = fig.colorbar(scatter00, ax = ax00)
                cbar00.set_label(label=c_label)
                cbar10 = fig.colorbar(scatter10, ax = ax10)
                cbar10.set_label(label=c_label)
                cbar20 = fig.colorbar(scatter20, ax = ax20)
                cbar20.set_label(label=c_label)

                cbar01 = fig.colorbar(scatter01, ax = ax01)
                cbar01.set_label(label=c_label)
                cbar11 = fig.colorbar(scatter11, ax = ax11)
                cbar11.set_label(label=c_label)
                cbar21 = fig.colorbar(scatter21, ax = ax21)
                cbar21.set_label(label=c_label)

                cbar02 = fig.colorbar(scatter02, ax = ax02)
                cbar02.set_label(label=c_label)
                cbar12 = fig.colorbar(scatter12, ax = ax12)
                cbar12.set_label(label=c_label)
                cbar22 = fig.colorbar(scatter22, ax = ax22)
                cbar22.set_label(label=c_label)

                cbar03 = fig.colorbar(scatter03, ax = ax03)
                cbar03.set_label(label=c_label)
                cbar13 = fig.colorbar(scatter13, ax = ax13)
                cbar13.set_label(label=c_label)
                cbar23 = fig.colorbar(scatter23, ax = ax23)
                cbar23.set_label(label=c_label)


            column_titles = ['LSQ', 'SEG', r'$\mathregular{DNN_{SL}}$', r'$\mathregular{DNN_{SSL}}$']
            pad = 50
            for ax, column_title in zip(axes[0], column_titles):
                ax.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', fontsize=40)


            for ax_set in axes.T:
                ax_set[0].plot(ref_Dt, ref_Dt, ls='--', c='black')
                ax_set[1].plot(ref_Fp, ref_Fp, ls='--', c='black')
                ax_set[2].plot(ref_Dp, ref_Dp, ls='--', c='black')
                
                ax_set[0].set(xlabel=r'GT $D_t$ [mm$^2$/s]', ylabel = r'Estimated $D_t$ [mm$^2$/s]')
                ax_set[1].set(xlabel=r'GT $f_p$', ylabel = r'Estimated $f_p$')
                ax_set[2].set(xlabel=r'GT $D_p$ [mm$^2$/s]', ylabel = r'Estimated $D_p$ [mm$^2$/s]')

                ax_set[0].yaxis.set_major_formatter(from_utils.OOMFormatter(-3, "%1.1f"))
                ax_set[2].yaxis.set_major_formatter(from_utils.OOMFormatter(-3, "%1.0f"))
                ax_set[0].xaxis.set_major_formatter(from_utils.OOMFormatter(-3, "%1.1f"))
                ax_set[2].xaxis.set_major_formatter(from_utils.OOMFormatter(-3, "%1.0f"))

            fig.tight_layout(pad=2.5)
            plt.savefig(os.path.join(self.dir_out,  f'predvstruth_SNR{snr}_{colordim}'))
            plt.savefig(os.path.join(self.dir_out,  f'predvstruth_SNR{snr}_{colordim}.png'))



    def plot_sim_stat(self):
        convert_dict_nrmse = {'SNR': str, 'param': object, 'method': object, 'param_val': float}
        convert_dict_rho = {'SNR': str, 'p': object, 'method': object, 'p_val': float}
        convert_dict_cv = {'SNR': str, 'param': object, 'method': object, 'cv_val': float}
        convert_dict_measured_signal_rmse = {'SNR': str, 'method': object, 'measured_signal_rmse_val': float}
        convert_dict_true_signal_rmse = {'SNR': str, 'method': object, 'true_signal_rmse_val': float}

        df_nrmse = pd.read_csv(os.path.join(self.dir_out, 'df_nrmse.csv'))
        df_rho = pd.read_csv(os.path.join(self.dir_out, 'df_rho.csv'))
        df_cv = pd.read_csv(os.path.join(self.dir_out, 'df_cv.csv'))
        df_measured_signal_rmse = pd.read_csv(os.path.join(self.dir_out, 'df_measured_signal_rmse.csv'))
        df_true_signal_rmse = pd.read_csv(os.path.join(self.dir_out, 'df_true_signal_rmse.csv'))

        df_nrmse = df_nrmse.astype(convert_dict_nrmse)
        df_rho = df_rho.astype(convert_dict_rho)
        df_cv = df_cv.astype(convert_dict_cv)
        df_measured_signal_rmse = df_measured_signal_rmse.astype(convert_dict_measured_signal_rmse)
        df_true_signal_rmse = df_true_signal_rmse.astype(convert_dict_true_signal_rmse)


        df_nrmse['SNR'] = pd.Categorical(df_nrmse['SNR'], categories=self.snrs)
        df_rho['SNR'] = pd.Categorical(df_rho['SNR'], categories=self.snrs)
        df_cv['SNR'] = pd.Categorical(df_cv['SNR'], categories=self.snrs)
        df_measured_signal_rmse['SNR'] = pd.Categorical(df_measured_signal_rmse['SNR'], categories=self.snrs)
        df_true_signal_rmse['SNR'] = pd.Categorical(df_true_signal_rmse['SNR'], categories=self.snrs)



        # plotting
        sns.set_theme()
        params = {'axes.labelsize': 35,
                  'axes.grid' : True, 
                  'axes.titlesize': 25,
                  'lines.markersize': 12.5,
                  'lines.linewidth': 3.75,
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25,
                  'legend.fontsize': 40, 
                  'legend.markerscale': 2.5,
                  'legend.loc': 'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        snrs_lables = [r'8', r'10', r'12', r'15', r'$\mathbf{20}$', r'25', r'35', r'50', r'75', r'100']
        snrs_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        estimator = np.mean
        errorbar=('ci', 99)


        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(22.5, 17.5))

        ax00 = sns.lineplot(data=df_nrmse[df_nrmse.param == 'Dt'], x='SNR', y='param_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[0,0])
        ax10 = sns.lineplot(data=df_nrmse[df_nrmse.param == 'Fp'], x='SNR', y='param_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[1,0])
        ax20 = sns.lineplot(data=df_nrmse[df_nrmse.param == 'Dp'], x='SNR', y='param_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[2,0])

        ax01 = sns.lineplot(data=df_rho[df_rho.p == 'DtDp'], x='SNR', y='p_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[0,1])
        ax11 = sns.lineplot(data=df_rho[df_rho.p == 'DtFp'], x='SNR', y='p_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[1,1])
        ax21 = sns.lineplot(data=df_rho[df_rho.p == 'DpFp'], x='SNR', y='p_val', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=['^', 'v', 's', 'D'], palette=['steelblue', 'coral', 'palevioletred', 'mediumseagreen'], dashes=False, ax=axes[2,1])

        ax02 = sns.lineplot(data=df_cv[df_cv.param == 'Dt'], x='SNR', y='cv_val', hue='method', estimator=estimator, style='method', markers=['s', 'D'], palette=['palevioletred', 'mediumseagreen'], dashes=False, ax=axes[0,2])
        ax12 = sns.lineplot(data=df_cv[df_cv.param == 'Fp'], x='SNR', y='cv_val', hue='method', estimator=estimator, style='method', markers=['s', 'D'], palette=['palevioletred', 'mediumseagreen'], dashes=False, ax=axes[1,2])
        ax22 = sns.lineplot(data=df_cv[df_cv.param == 'Dp'], x='SNR', y='cv_val', hue='method', estimator=estimator, style='method', markers=['s', 'D'], palette=['palevioletred', 'mediumseagreen'], dashes=False, ax=axes[2,2])
    
        ax00.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 0.5), ylabel = r'NMAE $D_t$')
        ax01.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 1), ylabel = r'$\left|\rho(D_t, D_p)\right|$')
        ax02.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 0.125), ylabel = r'CV $D_t$')

        ax10.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 0.75), ylabel = r'NMAE $f_p$')
        ax11.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 1), ylabel = r'$\left|\rho(f_p, D_t)\right|$')
        ax12.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 0.125), ylabel = r'CV $f_p$')

        ax20.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 2.25), ylabel = r'NMAE $D_p$')
        ax21.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 1), ylabel = r'$\left|\rho(D_p, f_p)\right|$')
        ax22.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(0, 0.125), ylabel = r'CV $D_p$')

        ax00.legend().set_visible(False); ax01.legend().set_visible(False); ax02.legend().set_visible(False)
        ax10.legend().set_visible(False); ax11.legend().set_visible(False); ax12.legend().set_visible(False)
        ax20.legend().set_visible(False); ax21.legend().set_visible(False); ax22.legend().set_visible(False)

        handles, labels = ax01.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

        for line in legend.get_lines():
            line.set_linewidth(10)  # Set line width


        fig.tight_layout(pad=2.5)
        plt.savefig(os.path.join(self.dir_out, f'stats_from_sims_NRMSE_p_CV.pdf'), bbox_inches='tight')    



    def print_sim_stats(self):
        df_nrmse = pd.read_csv(self.dir_out + '/df_nrmse.csv')
        df_rho = pd.read_csv(self.dir_out + '/df_rho.csv')
        df_cv = pd.read_csv(self.dir_out + '/df_cv.csv')
        df_measured_signal_rmse = pd.read_csv(self.dir_out + '/df_measured_signal_rmse.csv')
        df_true_signal_rmse = pd.read_csv(self.dir_out + '/df_true_signal_rmse.csv')

        nrmse_rows = ['Dt', 'Fp', 'Dp']
        rho_rows = ['DtDp', 'DtFp', 'DpFp']
        cv_rows = ['Dt', 'Fp', 'Dp']


        for snr in self.snrs:
            print('=====================================================')
            print(f'SNR: {snr}')


            print('*****************************************************')
            print('NMAE')
            for i in range(len(nrmse_rows)):

                nrmse_filtered_lsq = df_nrmse[(df_nrmse.SNR == int(snr)) & (df_nrmse.param == nrmse_rows[i])  & (df_nrmse.method == 'LSQ')]['param_val'].to_numpy()
                nrmse_filtered_seg = df_nrmse[(df_nrmse.SNR == int(snr)) & (df_nrmse.param == nrmse_rows[i])  & (df_nrmse.method == 'SEG')]['param_val'].to_numpy()
                nrmse_filtered_dnn_sup = df_nrmse[(df_nrmse.SNR == int(snr)) & (df_nrmse.param == nrmse_rows[i])  & (df_nrmse.method == r'$\mathregular{DNN_{SL}}$')]['param_val'].to_numpy()
                nrmse_filtered_dnn_slf = df_nrmse[(df_nrmse.SNR == int(snr)) & (df_nrmse.param == nrmse_rows[i])  & (df_nrmse.method == r'$\mathregular{DNN_{SSL}}$')]['param_val'].to_numpy()
                ci_nrmse_dnn_sup = from_utils.bootstrap_CI_99_mean(nrmse_filtered_dnn_sup)
                ci_nrmse_dnn_slf = from_utils.bootstrap_CI_99_mean(nrmse_filtered_dnn_slf)

                print('----------')
                print(nrmse_rows[i])
                print(f'LSQ: \t\t {nrmse_filtered_lsq[0]:.4f}')
                print(f'SEG: \t\t {nrmse_filtered_seg[0]:.4f}')
                print(f'DNN_SL: \t {np.mean(nrmse_filtered_dnn_sup):.4f},\t lower: {ci_nrmse_dnn_sup[0]:.4f} \t upper: {ci_nrmse_dnn_sup[1]:.4f}')
                print(f'DNN_SSL: \t {np.mean(nrmse_filtered_dnn_slf):.4f},\t lower: {ci_nrmse_dnn_slf[0]:.4f} \t upper: {ci_nrmse_dnn_slf[1]:.4f}')
                print('----------')



            print('*****************************************************')
            print('rho')
            for i in range(len(rho_rows)):
                rho_filtered_lsq = df_rho[(df_rho.SNR == int(snr)) & (df_rho.p == rho_rows[i])  & (df_rho.method == 'LSQ')]['p_val'].to_numpy()
                rho_filtered_seg = df_rho[(df_rho.SNR == int(snr)) & (df_rho.p == rho_rows[i])  & (df_rho.method == 'SEG')]['p_val'].to_numpy()
                rho_filtered_dnn_sup = df_rho[(df_rho.SNR == int(snr)) & (df_rho.p == rho_rows[i])  & (df_rho.method == r'$\mathregular{DNN_{SL}}$')]['p_val'].to_numpy()
                rho_filtered_dnn_slf = df_rho[(df_rho.SNR == int(snr)) & (df_rho.p == rho_rows[i])  & (df_rho.method == r'$\mathregular{DNN_{SSL}}$')]['p_val'].to_numpy()
                ci_rho_dnn_sup = from_utils.bootstrap_CI_99_mean(rho_filtered_dnn_sup)
                ci_rho_dnn_slf = from_utils.bootstrap_CI_99_mean(rho_filtered_dnn_slf)

                print('----------')
                print(rho_rows[i])
                print(f'LSQ: \t\t {rho_filtered_lsq[0]:.4f}')
                print(f'SEG: \t\t {rho_filtered_seg[0]:.4f}')
                print(f'DNN_SL: \t {np.mean(rho_filtered_dnn_sup):.4f},\t lower: {ci_rho_dnn_sup[0]:.4f} \t upper: {ci_rho_dnn_sup[1]:.4f}')
                print(f'DNN_SSL: \t {np.mean(rho_filtered_dnn_slf):.4f},\t lower: {ci_rho_dnn_slf[0]:.4f} \t upper: {ci_rho_dnn_slf[1]:.4f}')
                print('----------')



            print('*****************************************************')
            print('CV')
            for i in range(len(cv_rows)):
                cv_filtered_dnn_sup = df_cv[(df_cv.SNR == int(snr)) & (df_cv.param == cv_rows[i])  & (df_cv.method == r'$\mathregular{DNN_{SL}}$')]['cv_val'].to_numpy()
                cv_filtered_dnn_slf = df_cv[(df_cv.SNR == int(snr)) & (df_cv.param == cv_rows[i])  & (df_cv.method == r'$\mathregular{DNN_{SSL}}$')]['cv_val'].to_numpy()

                print('----------')
                print(cv_rows[i])
                print(f'DNN_SL: \t {np.mean(cv_filtered_dnn_sup):.4f}')
                print(f'DNN_SSL: \t {np.mean(cv_filtered_dnn_slf):.4f}')
                print('----------')
            


            print('*****************************************************')
            print('Estimated generalisation signal-RMSE')
            measured_signal_rmse_filtered_lsq = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == 'LSQ')]['measured_signal_rmse_val'].to_numpy()
            measured_signal_rmse_filtered_lsq = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == 'LSQ')]['measured_signal_rmse_val'].to_numpy()
            measured_signal_rmse_filtered_seg = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == 'SEG')]['measured_signal_rmse_val'].to_numpy()
            measured_signal_rmse_filtered_dnn_sup = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == r'$\mathregular{DNN_{SL}}$')]['measured_signal_rmse_val'].to_numpy()
            measured_signal_rmse_filtered_dnn_slf = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == r'$\mathregular{DNN_{SSL}}$')]['measured_signal_rmse_val'].to_numpy()
            measured_signal_rmse_filtered_gt = df_measured_signal_rmse[(df_measured_signal_rmse.SNR == int(snr))  & (df_measured_signal_rmse.method == 'GT')]['measured_signal_rmse_val'].to_numpy()
            ci_measured_signal_rmse_dnn_sup = from_utils.bootstrap_CI_99_mean(measured_signal_rmse_filtered_dnn_sup)
            ci_measured_signal_rmse_dnn_slf = from_utils.bootstrap_CI_99_mean(measured_signal_rmse_filtered_dnn_slf)

            print('----------')
            print(f'GT: \t\t {measured_signal_rmse_filtered_gt[0]:.4f}')
            print(f'LSQ: \t\t {measured_signal_rmse_filtered_lsq[0]:.4f}')
            print(f'SEG: \t\t {measured_signal_rmse_filtered_seg[0]:.4f}')
            print(f'DNN_SL: \t {np.mean(measured_signal_rmse_filtered_dnn_sup):.4f},\t lower: {ci_measured_signal_rmse_dnn_sup[0]:.4f} \t upper: {ci_measured_signal_rmse_dnn_sup[1]:.4f}')
            print(f'DNN_SSL: \t {np.mean(measured_signal_rmse_filtered_dnn_slf):.4f},\t lower: {ci_measured_signal_rmse_dnn_slf[0]:.4f} \t upper: {ci_measured_signal_rmse_dnn_slf[1]:.4f}')
            print('----------')



            print('*****************************************************')
            print('Estimated expected estimation signal-RMSE')

            true_signal_rmse_filtered_lsq = df_true_signal_rmse[(df_true_signal_rmse.SNR == int(snr))  & (df_true_signal_rmse.method == 'LSQ')]['true_signal_rmse_val'].to_numpy()
            true_signal_rmse_filtered_seg = df_true_signal_rmse[(df_true_signal_rmse.SNR == int(snr))  & (df_true_signal_rmse.method == 'SEG')]['true_signal_rmse_val'].to_numpy()
            true_signal_rmse_filtered_dnn_sup = df_true_signal_rmse[(df_true_signal_rmse.SNR == int(snr))  & (df_true_signal_rmse.method == r'$\mathregular{DNN_{SL}}$')]['true_signal_rmse_val'].to_numpy()
            true_signal_rmse_filtered_dnn_slf = df_true_signal_rmse[(df_true_signal_rmse.SNR == int(snr))  & (df_true_signal_rmse.method == r'$\mathregular{DNN_{SSL}}$')]['true_signal_rmse_val'].to_numpy()
            ci_true_signal_rmse_dnn_sup = from_utils.bootstrap_CI_99_mean(true_signal_rmse_filtered_dnn_sup)
            ci_true_signal_rmse_dnn_slf = from_utils.bootstrap_CI_99_mean(true_signal_rmse_filtered_dnn_slf)

            print('----------')
            print(f'LSQ: \t\t {true_signal_rmse_filtered_lsq[0]:.4f}')
            print(f'SEG: \t\t {true_signal_rmse_filtered_seg[0]:.4f}')
            print(f'DNN_SL: \t {np.mean(true_signal_rmse_filtered_dnn_sup):.4f},\t lower: {ci_true_signal_rmse_dnn_sup[0]:.4f} \t upper: {ci_true_signal_rmse_dnn_sup[1]:.4f}')
            print(f'DNN_SSL: \t {np.mean(true_signal_rmse_filtered_dnn_slf):.4f},\t lower: {ci_true_signal_rmse_dnn_slf[0]:.4f} \t upper: {ci_true_signal_rmse_dnn_slf[1]:.4f}')
            print('----------')      




class Simplt_allbvalsets:
    def __init__(self, snrs, reps, exp_name):
        self.snrs = [str(x) for x in snrs]
        self.reps = reps
        self.exp_name = exp_name

        #self.sim_bounds = np.array([[0.0005, 0.05, 0.005], [0.003, 0.50, 0.1]])
        #self.max = 800
        #self.bval_range_plt = np.linspace(0, self.max, self.max+1)

        self.dir_out = f'../simulations/simulations_plot'
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)


    def plt_signal_rmse(self, error):
        if error == 'estimated_generalisation_signal_rmse':
            convert_dict_signal_rmse = {'SNR': str, 'method': object, 'measured_signal_rmse_val': float}
            df_name = 'measured_signal_rmse'
            y = 'measured_signal_rmse_val'
            y_label = 'observed-signal-RMSE'
            palette = ['darkgoldenrod', 'steelblue', 'coral', 'palevioletred', 'mediumseagreen']
            markers=['o', '^', 'v', 's', 'D']
            n_legend_cols = 5
        elif error == 'estimated_expected_estimation_signal_rmse':
            convert_dict_signal_rmse = {'SNR': str, 'method': object, 'true_signal_rmse_val': float}
            df_name = 'true_signal_rmse'
            y = 'true_signal_rmse_val'
            y_label = 'true-signal-RMSE'
            palette = ['steelblue', 'coral', 'palevioletred', 'mediumseagreen']
            markers=['^', 'v', 's', 'D']
            n_legend_cols = 4


        df_signal_rmse_b11 = pd.read_csv(os.path.join(self.dir_out,  f'b11/{self.exp_name}/df_{df_name}.csv'))
        df_signal_rmse_b5 = pd.read_csv(os.path.join(self.dir_out,  f'b5/{self.exp_name}/df_{df_name}.csv'))
        df_signal_rmse_b4 = pd.read_csv(os.path.join(self.dir_out,  f'b4/{self.exp_name}/df_{df_name}.csv'))

        df_signal_rmse_b11 = df_signal_rmse_b11.astype(convert_dict_signal_rmse)
        df_signal_rmse_b5 = df_signal_rmse_b5.astype(convert_dict_signal_rmse)
        df_signal_rmse_b4 = df_signal_rmse_b4.astype(convert_dict_signal_rmse)

        df_signal_rmse_b11['SNR'] = pd.Categorical(df_signal_rmse_b11['SNR'], categories=self.snrs)
        df_signal_rmse_b5['SNR'] = pd.Categorical(df_signal_rmse_b5['SNR'], categories=self.snrs)
        df_signal_rmse_b4['SNR'] = pd.Categorical(df_signal_rmse_b4['SNR'], categories=self.snrs)


        # plotting
        sns.set_theme()
        params = {'axes.labelsize': 35,
                  'axes.grid' : True, 
                  'axes.titlesize': 45,
                  'lines.markersize': 12.5,
                  'lines.linewidth': 3.75,
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 25,
                  'legend.fontsize': 40, 
                  'legend.markerscale': 2.5,
                  'legend.loc': 'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        snrs_lables = [r'8', r'10', r'12', r'15', r'$\mathbf{20}$', r'25', r'35', r'50', r'75', r'100']
        snrs_pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        estimator = np.mean
        errorbar=('ci', 99)

        fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (24, 9))
        [ax0, ax1, ax2] = axes

        ax0 = sns.lineplot(ax=ax0, data=df_signal_rmse_b11, x='SNR', y=f'{y}', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=markers, palette=palette, dashes=False)
        ax1 = sns.lineplot(ax=ax1, data=df_signal_rmse_b5, x='SNR', y=f'{y}', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=markers, palette=palette, dashes=False)
        ax2 = sns.lineplot(ax=ax2, data=df_signal_rmse_b4, x='SNR', y=f'{y}', hue='method', estimator=estimator, errorbar=errorbar, style='method', markers=markers, palette=palette, dashes=False)

        ax0.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(-0.005, 0.09), ylabel=y_label)
        ax1.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(-0.005, 0.09), ylabel=y_label)
        ax2.set(xticks=snrs_pos, xticklabels=snrs_lables, ylim=(-0.005, 0.09), ylabel=y_label)
        ax0.set_title(label=r'11 b-values', pad=15)
        ax1.set_title(label=r'5 b-values', pad=15)
        ax2.set_title(label=r'4 b-values', pad=15)

        ax0.legend().set_visible(False); ax1.legend().set_visible(False); ax2.legend().set_visible(False)

        handles, labels = ax0.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=n_legend_cols, fontsize=30)

        for line in legend.get_lines():
            line.set_linewidth(10)

        fig.tight_layout(pad=5)
        plt.savefig(os.path.join(self.dir_out, f'{error}.pdf'), bbox_inches='tight') 
