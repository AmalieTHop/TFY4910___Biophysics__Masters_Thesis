"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

import os

from matplotlib.ticker import FormatStrFormatter
from tueplots import fonts
from paretoset import paretoset





class BO_plt:
    def __init__(self):
        self.dir_out = f'../simulations/simulations_plot'
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)
        
        # user specific input: model selection criteria (appendix A)
        self.sum_nmae = 1
        self.Fp_nmae = 0.24454
        self.Dt_nmae = 0.19085
        self.Dp_nmae = 0.66299

        # user specific input: max and min value of x- and y-axis in plots
        self.xmin_slf_b4, self.xmax_slf_b4 = 0.016, 0.0225
        self.ymin_slf_b4, self.ymax_slf_b4 = 0.825, 1.025
        self.xmin_sup_b4, self.xmax_sup_b4 = 0.030, 0.033
        self.ymin_sup_b4, self.ymax_sup_b4 = 0.758, 0.762 
        self.xmin_slf_b5, self.xmax_slf_b5 = 0.023, 0.0295
        self.ymin_slf_b5, self.ymax_slf_b5 = 0.82, 0.96
        self.xmin_sup_b5, self.xmax_sup_b5 = 0.034, 0.036
        self.ymin_sup_b5, self.ymax_sup_b5 = 0.727, 0.730
        self.xmin_slf_b11, self.xmax_slf_b11 = 0.039, 0.043
        self.ymin_slf_b11, self.ymax_slf_b11 = 0.65, 0.90
        self.xmin_sup_b11, self.xmax_sup_b11 = 0.050, 0.056
        self.ymin_sup_b11, self.ymax_sup_b11 = 0.598, 0.61

        # user specific input: model selection
        self.expid_sup_b11 = 14
        self.expid_sup_b5 = 148
        self.expid_sup_b4 = 150
        self.expid_slf_b11 = 72
        self.expid_slf_b5 = 133
        self.expid_slf_b4 = 26 



        # full analysis
        self.df_BO_full_sup_b11 = pd.read_csv(f'../simulations/BO/b11/snr20/dnn_sup/df_trials_249.csv')
        self.df_BO_full_sup_b5 = pd.read_csv(f'../simulations/BO/b5/snr20/dnn_sup/df_trials_249.csv')
        self.df_BO_full_sup_b4 = pd.read_csv(f'../simulations/BO/b4/snr20/dnn_sup/df_trials_249.csv')
        self.df_BO_full_slf_b11 = pd.read_csv(f'../simulations/BO/b11/snr20/dnn_slf/df_trials_249.csv')
        self.df_BO_full_slf_b5 = pd.read_csv(f'../simulations/BO/b5/snr20/dnn_slf/df_trials_249.csv')
        self.df_BO_full_slf_b4 = pd.read_csv(f'../simulations/BO/b4/snr20/dnn_slf/df_trials_249.csv')

        # reduced analysis: nmae-sum and signal-rmse on val_MS
        self.df_BO_redu_sup_b11 = self.df_BO_full_sup_b11[["sum_nmae_valMS", "loss_valMS"]]
        self.df_BO_redu_sup_b5 = self.df_BO_full_sup_b5[["sum_nmae_valMS", "loss_valMS"]]
        self.df_BO_redu_sup_b4 = self.df_BO_full_sup_b4[["sum_nmae_valMS", "loss_valMS"]]
        self.df_BO_redu_slf_b11 = self.df_BO_full_slf_b11[["sum_nmae_valMS", "loss_valMS"]]
        self.df_BO_redu_slf_b5 = self.df_BO_full_slf_b5[["sum_nmae_valMS", "loss_valMS"]]
        self.df_BO_redu_slf_b4 = self.df_BO_full_slf_b4[["sum_nmae_valMS", "loss_valMS"]]
    


    def plot_BO_all_in_one(self):
        """
        All BO experiments from all DNN models are plotted in one plot.
        (Figure 4.1)
        """

        # pareto frontier
        df_BO_pareto_bool_sup_b11 = paretoset(self.df_BO_redu_sup_b11, sense=["min", "min"])
        df_BO_pareto_bool_sup_b5 = paretoset(self.df_BO_redu_sup_b5, sense=["min", "min"])
        df_BO_pareto_bool_sup_b4 = paretoset(self.df_BO_redu_sup_b4, sense=["min", "min"])
        df_BO_pareto_bool_slf_b11 = paretoset(self.df_BO_redu_slf_b11, sense=["min", "min"])
        df_BO_pareto_bool_slf_b5 = paretoset(self.df_BO_redu_slf_b5, sense=["min", "min"])
        df_BO_pareto_bool_slf_b4 = paretoset(self.df_BO_redu_slf_b4, sense=["min", "min"])

        # full analysis of pareto frontier
        df_BO_pareto_full_sup_b11 = self.df_BO_full_sup_b11[df_BO_pareto_bool_sup_b11]
        df_BO_pareto_full_sup_b5 = self.df_BO_full_sup_b5[df_BO_pareto_bool_sup_b5]
        df_BO_pareto_full_sup_b4 = self.df_BO_full_sup_b4[df_BO_pareto_bool_sup_b4]
        df_BO_pareto_full_slf_b11 = self.df_BO_full_slf_b11[df_BO_pareto_bool_slf_b11]
        df_BO_pareto_full_slf_b5 = self.df_BO_full_slf_b5[df_BO_pareto_bool_slf_b5]
        df_BO_pareto_full_slf_b4 = self.df_BO_full_slf_b4[df_BO_pareto_bool_slf_b4]

        # reduced analysis of pareto frontier: nmae-sum and signal-rmse on val_MS
        df_BO_pareto_redu_sup_b11 = self.df_BO_redu_sup_b11[df_BO_pareto_bool_sup_b11]
        df_BO_pareto_redu_sup_b5 = self.df_BO_redu_sup_b5[df_BO_pareto_bool_sup_b5]
        df_BO_pareto_redu_sup_b4 = self.df_BO_redu_sup_b4[df_BO_pareto_bool_sup_b4]
        df_BO_pareto_redu_slf_b11 = self.df_BO_redu_slf_b11[df_BO_pareto_bool_slf_b11]
        df_BO_pareto_redu_slf_b5 = self.df_BO_redu_slf_b5[df_BO_pareto_bool_slf_b5]
        df_BO_pareto_redu_slf_b4 = self.df_BO_redu_slf_b4[df_BO_pareto_bool_slf_b4]

        # model selection criteria (appendix A)
        df_BO_pareto_filtered_full_sup_b11 = df_BO_pareto_full_sup_b11[(self.df_BO_full_sup_b11['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                       (self.df_BO_full_sup_b11['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                       (self.df_BO_full_sup_b11['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                       (self.df_BO_full_sup_b11['sum_nmae_valMS'] <= self.sum_nmae)]
        df_BO_pareto_filtered_full_sup_b5 = df_BO_pareto_full_sup_b5[(self.df_BO_full_sup_b5['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                     (self.df_BO_full_sup_b5['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                     (self.df_BO_full_sup_b5['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                     (self.df_BO_full_sup_b5['sum_nmae_valMS'] <= self.sum_nmae)]
        df_BO_pareto_filtered_full_sup_b4 = df_BO_pareto_full_sup_b4[(self.df_BO_full_sup_b4['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                     (self.df_BO_full_sup_b4['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                     (self.df_BO_full_sup_b4['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                     (self.df_BO_full_sup_b4['sum_nmae_valMS'] <= self.sum_nmae)]
        df_BO_pareto_filtered_full_slf_b11 = df_BO_pareto_full_slf_b11[(self.df_BO_full_slf_b11['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                       (self.df_BO_full_slf_b11['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                       (self.df_BO_full_slf_b11['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                       (self.df_BO_full_slf_b11['sum_nmae_valMS'] <= self.sum_nmae)]
        df_BO_pareto_filtered_full_slf_b5 = df_BO_pareto_full_slf_b5[(self.df_BO_full_slf_b5['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                     (self.df_BO_full_slf_b5['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                     (self.df_BO_full_slf_b5['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                     (self.df_BO_full_slf_b5['sum_nmae_valMS'] <= self.sum_nmae)]
        df_BO_pareto_filtered_full_slf_b4 = df_BO_pareto_full_slf_b4[(self.df_BO_full_slf_b4['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                                     (self.df_BO_full_slf_b4['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                                     (self.df_BO_full_slf_b4['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                                     (self.df_BO_full_slf_b4['sum_nmae_valMS'] <= self.sum_nmae)]



        # plotting
        sns.set_theme()
        params = {'axes.titlesize': 70,
                  'axes.labelsize': 55,
                  'axes.grid' : True, 
                  'xtick.labelsize': 50,
                  'ytick.labelsize': 50,
                  'lines.markersize': 15,
                  'legend.fontsize': 60, 
                  'legend.framealpha': 0.50,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        s_star = 5000
        s_pareto = 500


        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 40))
        [(ax00, ax01), (ax10, ax11), (ax20, ax21)] = axes
        

        ax00 = sns.scatterplot(ax=ax00, data=self.df_BO_redu_sup_b11, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax00 = sns.scatterplot(ax=ax00, data=df_BO_pareto_redu_sup_b11, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax00 = sns.scatterplot(ax=ax00, data=df_BO_pareto_filtered_full_sup_b11, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax00 = sns.scatterplot(ax=ax00, data=self.df_BO_redu_sup_b11[self.expid_sup_b11:self.expid_sup_b11+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax00.set(xlim=(self.xmin_sup_b11, self.xmax_sup_b11), ylim=(self.ymin_sup_b11, self.ymax_sup_b11), title=r'$\mathregular{DNN_{SL}^{11b}}$')
        ax00.set_title(label=r'$\mathregular{DNN_{SL}^{11b}}$', pad=20)

        ax10 = sns.scatterplot(ax=ax10, data=self.df_BO_redu_sup_b5, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax10 = sns.scatterplot(ax=ax10, data=df_BO_pareto_redu_sup_b5, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax10 = sns.scatterplot(ax=ax10, data=df_BO_pareto_filtered_full_sup_b5, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax10 = sns.scatterplot(ax=ax10, data=self.df_BO_redu_sup_b5[self.expid_sup_b5:self.expid_sup_b5+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax10.set(xlim=(self.xmin_sup_b5, self.xmax_sup_b5), ylim=(self.ymin_sup_b5, self.ymax_sup_b5), title=r'$\mathregular{DNN_{SL}^{5b}}$')
        ax10.set_title(label=r'$\mathregular{DNN_{SL}^{5b}}$', pad=20)

        ax20 = sns.scatterplot(ax=ax20, data=self.df_BO_redu_sup_b4, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax20 = sns.scatterplot(ax=ax20, data=df_BO_pareto_redu_sup_b4, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax20 = sns.scatterplot(ax=ax20, data=df_BO_pareto_filtered_full_sup_b4, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax20 = sns.scatterplot(ax=ax20, data=self.df_BO_redu_sup_b4[self.expid_sup_b4:self.expid_sup_b4+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax20.set(xlim=(self.xmin_sup_b4, self.xmax_sup_b4), ylim=(self.ymin_sup_b4, self.ymax_sup_b4), title=r'$\mathregular{DNN_{SL}^{4b}}$')
        ax20.set_title(label=r'$\mathregular{DNN_{SL}^{4b}}$', pad=20)

        ax01 = sns.scatterplot(ax=ax01, data=self.df_BO_redu_slf_b11, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax01 = sns.scatterplot(ax=ax01, data=df_BO_pareto_redu_slf_b11, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax01 = sns.scatterplot(ax=ax01, data=df_BO_pareto_filtered_full_slf_b11, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax01 = sns.scatterplot(ax=ax01, data=self.df_BO_redu_slf_b11[self.expid_slf_b11:self.expid_slf_b11+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax01.set(xlim=(self.xmin_slf_b11, self.xmax_slf_b11), ylim=(self.ymin_slf_b11, self.ymax_slf_b11))
        ax01.set_title(label=r'$\mathregular{DNN_{SSL}^{11b}}$', pad=20)

        ax11 = sns.scatterplot(ax=ax11, data=self.df_BO_redu_slf_b5, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax11 = sns.scatterplot(ax=ax11, data=df_BO_pareto_redu_slf_b5, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax11 = sns.scatterplot(ax=ax11, data=df_BO_pareto_filtered_full_slf_b5, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax11 = sns.scatterplot(ax=ax11, data=self.df_BO_redu_slf_b5[self.expid_slf_b5:self.expid_slf_b5+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax11.set(xlim=(self.xmin_slf_b5, self.xmax_slf_b5), ylim=(self.ymin_slf_b5, self.ymax_slf_b5))
        ax11.set_title(label=r'$\mathregular{DNN_{SSL}^{5b}}$', pad=20)

        ax21 = sns.scatterplot(ax=ax21, data=self.df_BO_redu_slf_b4, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        ax21 = sns.scatterplot(ax=ax21, data=df_BO_pareto_redu_slf_b4, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato', s=s_pareto)
        ax21 = sns.scatterplot(ax=ax21, data=df_BO_pareto_filtered_full_slf_b4, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax21 = sns.scatterplot(ax=ax21, data=self.df_BO_redu_slf_b4[self.expid_slf_b4:self.expid_slf_b4+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax21.set(xlim=(self.xmin_slf_b4, self.xmax_slf_b4), ylim=(self.ymin_slf_b4, self.ymax_slf_b4))
        ax21.set_title(label=r'$\mathregular{DNN_{SSL}^{4b}}$', pad=20)

        handles, labels = ax01.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=1)
        for i in range(4):
            legend.legendHandles[i]._sizes = [2500]
            if i==3:
                legend.legendHandles[i]._sizes = [5000]

        for ax in axes.flatten():
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.set(xlabel=r'signal-RMSE', ylabel=r'NMAE-SUM')
            ax.legend().set_visible(False)


        fig.tight_layout(pad=5)
        plt.savefig(os.path.join(self.dir_out, f'BO_all_individual_models.pdf'), bbox_inches='tight')
        




    def plot_BO_individual(self, num_bvals, learning):
        """
        All BO experiments for the DNN model specificed by 'num_bvals' and 'learning' is plotted.
        """

        dir_out = f'../simulations/simulations_plot/b{num_bvals}'
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # pretty prinitng the max and min of the axis
        if num_bvals == 4:
            if learning == 'slf':
                xmin, xmax = self.xmin_slf_b4, self.xmax_slf_b4
                ymin, ymax = self.ymin_slf_b4, self.ymax_slf_b4
            elif learning == 'sup':
                xmin, xmax = self.xmin_sup_b4, self.xmax_sup_b4
                ymin, ymax = self.ymin_sup_b4, self.ymax_sup_b4
        elif num_bvals == 5:
            if learning == 'slf':
                xmin, xmax = self.xmin_slf_b5, self.xmax_slf_b5
                ymin, ymax = self.ymin_slf_b5, self.ymax_slf_b5
            elif learning == 'sup':
                xmin, xmax = self.xmin_sup_b5, self.xmax_sup_b5
                ymin, ymax = self.ymin_sup_b5, self.ymax_sup_b5
        elif num_bvals == 11:
            if learning == 'slf':
                xmin, xmax = self.xmin_slf_b11, self.xmax_slf_b11
                ymin, ymax = self.ymin_slf_b11, self.ymax_slf_b11
            elif learning == 'sup':
                xmin, xmax = self.xmin_sup_b11, self.xmax_sup_b11
                ymin, ymax = self.ymin_sup_b11, self.ymax_sup_b11


        # full analysis
        df_BO_full = pd.read_csv(f'../simulations/bo/b{num_bvals}/snr20/dnn_{learning}/df_trials_249.csv')
        
        # reduced analysis: nmae-sum and signal-rmse on val_MS
        df_BO_redu = df_BO_full[["sum_nmae_valMS", "loss_valMS"]]

        # pareto frontier
        df_BO_pareto_bool = paretoset(df_BO_redu, sense=["min", "min"])

        # full analysis of pareto frontier
        df_BO_pareto_full = df_BO_full[df_BO_pareto_bool]

        # reduced analysis of pareto frontier: nmae-sum and signal-rmse on val_MS
        df_BO_pareto_redu = df_BO_redu[df_BO_pareto_bool]

        # model selection criteria (appendix A)
        pareto_filtered_df_BO = df_BO_pareto_full[(df_BO_full['Dt_nmae_valMS'] <= self.Dt_nmae) & 
                                                  (df_BO_full['Fp_nmae_valMS'] <= self.Fp_nmae) &
                                                  (df_BO_full['Dp_nmae_valMS'] <= self.Dp_nmae) &
                                                  (df_BO_full['sum_nmae_valMS'] <= self.sum_nmae)]
        
        # sort based on learning domain
        if learning == 'slf':
            sorted_pareto_filtered_df_BO = pareto_filtered_df_BO.sort_values('loss_valMS')
        elif learning == 'sup':
            sorted_pareto_filtered_df_BO = pareto_filtered_df_BO.sort_values('sum_nmae_valMS')



        # plotting
        sns.set_theme()
        params = {'axes.labelsize': 60,
                  'axes.grid' : True, 
                  'xtick.labelsize': 40,
                  'ytick.labelsize': 40,
                  'lines.markersize': 20,
                  'legend.fontsize': 50, 
                  'legend.framealpha': 0.50,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)


        fig, ax = plt.subplots(figsize=(20, 15))


        # exteriments
        ax = sns.scatterplot(data=df_BO_redu, x='loss_valMS', y='sum_nmae_valMS', label='Experiment', c='tan')
        
        # pareto experiments
        if learning == 'slf' and num_bvals != 11:
            ax = sns.scatterplot(data=df_BO_pareto_redu, x='loss_valMS', y='sum_nmae_valMS', label='Pareto experiment', c='tomato')
        
        # pareto experiemnts fulfilling the model selection criterias
        ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO, x='loss_valMS', y='sum_nmae_valMS', label='Passed pareto experiment', c='firebrick')
        
        # selected experiment as "best"
        if learning == 'slf':
            if num_bvals == 11:
                ax = sns.scatterplot(data=df_BO_redu[self.expid_slf_b11:self.expid_slf_b11+1], x='loss_valMS', y='sum_nmae_valMS', label='Selected passed pareto experiment', c='dodgerblue', marker='*', s=5000)
            else:
                ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO[:1], x='loss_valMS', y='sum_nmae_valMS', label='Selected passed pareto experiment', c='dodgerblue', marker='*', s=5000)
        elif learning == 'sup':
            ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO[:1], x='loss_valMS', y='sum_nmae_valMS', label='Selected passed pareto experiment', c='dodgerblue', marker='*', s=5000)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.xaxis.set_ticks(np.arange(xmin, xmax + 0.001, 0.001))
        if learning == 'sup':
            ax.yaxis.set_ticks(np.arange(ymin, ymax, 0.001))

        ax.set(xlabel=r'Signal-RMSE', ylabel=r'NMAE-SUM')
        if learning == 'slf':
            ax.legend(title=None, loc='upper right')
        elif learning == 'sup':
            ax.legend(title=None, loc='upper left')

        fig.tight_layout()
        plt.savefig(os.path.join(dir_out, f'BO_b{num_bvals}_{learning}.pdf'))    





    def plot_BO_all_individuals(self):
        """
        One figure with six subplots, where each subplots contain the BO experiments of a specific DNN model. 
        (Figure 4.2)
        """

        # plotting
        sns.set_theme()
        params = {'axes.labelsize': 75,
                  'axes.grid' : True, 
                  'xtick.labelsize': 50,
                  'ytick.labelsize': 50,
                  'lines.markersize': 15,
                  'legend.fontsize': 50, 
                  'legend.framealpha': 0.50,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        colors_cb = sns.color_palette('colorblind')
        colors_b = sns.color_palette('bright')


        fig, ax = plt.subplots(figsize=(30, 15))

        ax = sns.scatterplot(data=self.df_BO_redu_sup_b11, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SL}^{11b}}$', c=[colors_cb[3]])
        ax = sns.scatterplot(data=self.df_BO_redu_sup_b5, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SL}^{5b}}$', c=[colors_cb[1]])
        ax = sns.scatterplot(data=self.df_BO_redu_sup_b4, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SL}^{4b}}$', c=[colors_cb[8]])
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b11, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SSL}^{11b}}$', c=[colors_cb[2]])
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b5, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SSL}^{5b}}$', c=[colors_cb[0]])
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b4, x='loss_valMS', y='sum_nmae_valMS', label=r'$\mathregular{DNN_{SSL}^{4b}}$', c=[colors_cb[4]])

        ax = sns.scatterplot(data=self.df_BO_redu_sup_b11[self.expid_sup_b11:self.expid_sup_b11+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[3]], s=2500)
        ax = sns.scatterplot(data=self.df_BO_redu_sup_b5[self.expid_sup_b5:self.expid_sup_b5+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[1]], s=2500)
        ax = sns.scatterplot(data=self.df_BO_redu_sup_b4[self.expid_sup_b4:self.expid_sup_b4+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[8]], s=2500)
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b11[self.expid_slf_b11:self.expid_slf_b11+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[2]], s=2500)
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b5[self.expid_slf_b5:self.expid_slf_b5+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[0]], s=2500)
        ax = sns.scatterplot(data=self.df_BO_redu_slf_b4[self.expid_slf_b4:self.expid_slf_b4+1], x='loss_valMS', y='sum_nmae_valMS', marker='*', c=[colors_b[4]], s=2500)

        ax.set(xlabel=r'signal-RMSE', ylabel=r'NMAE-SUM')
        ax.set_xlim(0.01, 0.062)
        ax.set_ylim(0.58, 1.9)

        lgnd = plt.legend()
        for i in range(6):
            lgnd.legendHandles[i]._sizes = [2500]

        fig.tight_layout()
        plt.savefig(os.path.join(self.dir_out, f'BO_all_models.pdf'))    