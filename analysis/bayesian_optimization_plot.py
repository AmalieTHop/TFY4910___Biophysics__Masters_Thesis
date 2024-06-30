import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

import os

from matplotlib.ticker import FormatStrFormatter
from tueplots import figsizes, fonts
from paretoset import paretoset





class BOplt:
    def __init__(self):
        self.dir_out = f'../simulations/simulations_plot'
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)
        
        self.sum_nmae = 1
        self.Fp_nmae = 0.24454
        self.Dt_nmae = 0.19085
        self.Dp_nmae = 0.66299

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
    


    def plot_BO_individual_models_allinone(self):
        df_BO_sup_b11 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b11_sup.csv')
        df_BO_sup_b5 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b5_sup.csv')
        df_BO_sup_b4 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b4_sup.csv')
        df_BO_slf_b11 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b11_slf.csv')
        df_BO_slf_b5 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b5_slf.csv')
        df_BO_slf_b4 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b4_slf.csv')

        df_BO_valHTnmae_sup_b11 = df_BO_sup_b11[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_sup_b5 = df_BO_sup_b5[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_sup_b4 = df_BO_sup_b4[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b11 = df_BO_slf_b11[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b5 = df_BO_slf_b5[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b4 = df_BO_slf_b4[["sum_nmae_valHT", "loss_valHT"]]

        bool_pareto_df_BO_sup_b11 = paretoset(df_BO_valHTnmae_sup_b11, sense=["min", "min"])
        bool_pareto_df_BO_sup_b5 = paretoset(df_BO_valHTnmae_sup_b5, sense=["min", "min"])
        bool_pareto_df_BO_sup_b4 = paretoset(df_BO_valHTnmae_sup_b4, sense=["min", "min"])
        bool_pareto_df_BO_slf_b11 = paretoset(df_BO_valHTnmae_slf_b11, sense=["min", "min"])
        bool_pareto_df_BO_slf_b5 = paretoset(df_BO_valHTnmae_slf_b5, sense=["min", "min"])
        bool_pareto_df_BO_slf_b4 = paretoset(df_BO_valHTnmae_slf_b4, sense=["min", "min"])

        pareto_df_BO_sup_b11 = df_BO_valHTnmae_sup_b11[bool_pareto_df_BO_sup_b11]
        pareto_df_BO_sup_b5 = df_BO_valHTnmae_sup_b5[bool_pareto_df_BO_sup_b5]
        pareto_df_BO_sup_b4 = df_BO_valHTnmae_sup_b4[bool_pareto_df_BO_sup_b4]
        pareto_df_BO_slf_b11 = df_BO_valHTnmae_slf_b11[bool_pareto_df_BO_slf_b11]
        pareto_df_BO_slf_b5 = df_BO_valHTnmae_slf_b5[bool_pareto_df_BO_slf_b5]
        pareto_df_BO_slf_b4 = df_BO_valHTnmae_slf_b4[bool_pareto_df_BO_slf_b4]

        df_BO_pareto_sup_b11 = df_BO_sup_b11[bool_pareto_df_BO_sup_b11]
        df_BO_pareto_sup_b5 = df_BO_sup_b5[bool_pareto_df_BO_sup_b5]
        df_BO_pareto_sup_b4 = df_BO_sup_b4[bool_pareto_df_BO_sup_b4]
        df_BO_pareto_slf_b11 = df_BO_slf_b11[bool_pareto_df_BO_slf_b11]
        df_BO_pareto_slf_b5 = df_BO_slf_b5[bool_pareto_df_BO_slf_b5]
        df_BO_pareto_slf_b4 = df_BO_slf_b4[bool_pareto_df_BO_slf_b4]

        pareto_filtered_df_BO_sup_b11 = df_BO_pareto_sup_b11[(df_BO_sup_b11['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                             (df_BO_sup_b11['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                             (df_BO_sup_b11['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                             (df_BO_sup_b11['sum_nmae_valHT'] <= self.sum_nmae)]
        pareto_filtered_df_BO_sup_b5 = df_BO_pareto_sup_b5[(df_BO_sup_b5['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                           (df_BO_sup_b5['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                           (df_BO_sup_b5['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                           (df_BO_sup_b5['sum_nmae_valHT'] <= self.sum_nmae)]
        pareto_filtered_df_BO_sup_b4 = df_BO_pareto_sup_b4[(df_BO_sup_b4['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                           (df_BO_sup_b4['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                           (df_BO_sup_b4['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                           (df_BO_sup_b4['sum_nmae_valHT'] <= self.sum_nmae)]
        pareto_filtered_df_BO_slf_b11 = df_BO_pareto_slf_b11[(df_BO_slf_b11['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                             (df_BO_slf_b11['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                             (df_BO_slf_b11['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                             (df_BO_slf_b11['sum_nmae_valHT'] <= self.sum_nmae)]
        pareto_filtered_df_BO_slf_b5 = df_BO_pareto_slf_b5[(df_BO_slf_b5['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                           (df_BO_slf_b5['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                           (df_BO_slf_b5['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                           (df_BO_slf_b5['sum_nmae_valHT'] <= self.sum_nmae)]
        pareto_filtered_df_BO_slf_b4 = df_BO_pareto_slf_b4[(df_BO_slf_b4['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                                                           (df_BO_slf_b4['Fp_nmae_valHT'] <= self.Fp_nmae) &
                                                           (df_BO_slf_b4['Dp_nmae_valHT'] <= self.Dp_nmae) &
                                                           (df_BO_slf_b4['sum_nmae_valHT'] <= self.sum_nmae)]


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
        

        ax00 = sns.scatterplot(ax=ax00, data=df_BO_valHTnmae_sup_b11, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax00 = sns.scatterplot(ax=ax00, data=pareto_df_BO_sup_b11, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax00 = sns.scatterplot(ax=ax00, data=pareto_filtered_df_BO_sup_b11, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax00 = sns.scatterplot(ax=ax00, data=df_BO_valHTnmae_sup_b11[14:15], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax00.set(xlim=(self.xmin_sup_b11, self.xmax_sup_b11), ylim=(self.ymin_sup_b11, self.ymax_sup_b11), title=r'$\mathregular{DNN_{SL}^{11b}}$')
        ax00.set_title(label=r'$\mathregular{DNN_{SL}^{11b}}$', pad=20)

        ax10 = sns.scatterplot(ax=ax10, data=df_BO_valHTnmae_sup_b5, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax10 = sns.scatterplot(ax=ax10, data=pareto_df_BO_sup_b5, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax10 = sns.scatterplot(ax=ax10, data=pareto_filtered_df_BO_sup_b5, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax10 = sns.scatterplot(ax=ax10, data=df_BO_valHTnmae_sup_b5[148:149], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax10.set(xlim=(self.xmin_sup_b5, self.xmax_sup_b5), ylim=(self.ymin_sup_b5, self.ymax_sup_b5), title=r'$\mathregular{DNN_{SL}^{5b}}$')
        ax10.set_title(label=r'$\mathregular{DNN_{SL}^{5b}}$', pad=20)

        ax20 = sns.scatterplot(ax=ax20, data=df_BO_valHTnmae_sup_b4, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax20 = sns.scatterplot(ax=ax20, data=pareto_df_BO_sup_b4, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax20 = sns.scatterplot(ax=ax20, data=pareto_filtered_df_BO_sup_b4, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax20 = sns.scatterplot(ax=ax20, data=df_BO_valHTnmae_sup_b4[150:151], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax20.set(xlim=(self.xmin_sup_b4, self.xmax_sup_b4), ylim=(self.ymin_sup_b4, self.ymax_sup_b4), title=r'$\mathregular{DNN_{SL}^{4b}}$')
        ax20.set_title(label=r'$\mathregular{DNN_{SL}^{4b}}$', pad=20)

        ax01 = sns.scatterplot(ax=ax01, data=df_BO_valHTnmae_slf_b11, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax01 = sns.scatterplot(ax=ax01, data=pareto_df_BO_slf_b11, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax01 = sns.scatterplot(ax=ax01, data=pareto_filtered_df_BO_slf_b11, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax01 = sns.scatterplot(ax=ax01, data=df_BO_valHTnmae_slf_b11[72:73], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax01.set(xlim=(self.xmin_slf_b11, self.xmax_slf_b11), ylim=(self.ymin_slf_b11, self.ymax_slf_b11))
        ax01.set_title(label=r'$\mathregular{DNN_{SSL}^{11b}}$', pad=20)

        ax11 = sns.scatterplot(ax=ax11, data=df_BO_valHTnmae_slf_b5, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax11 = sns.scatterplot(ax=ax11, data=pareto_df_BO_slf_b5, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax11 = sns.scatterplot(ax=ax11, data=pareto_filtered_df_BO_slf_b5, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax11 = sns.scatterplot(ax=ax11, data=df_BO_valHTnmae_slf_b5[133:134], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
        ax11.set(xlim=(self.xmin_slf_b5, self.xmax_slf_b5), ylim=(self.ymin_slf_b5, self.ymax_slf_b5))
        ax11.set_title(label=r'$\mathregular{DNN_{SSL}^{5b}}$', pad=20)

        ax21 = sns.scatterplot(ax=ax21, data=df_BO_valHTnmae_slf_b4, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        ax21 = sns.scatterplot(ax=ax21, data=pareto_df_BO_slf_b4, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato', s=s_pareto)
        ax21 = sns.scatterplot(ax=ax21, data=pareto_filtered_df_BO_slf_b4, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick', s=s_pareto)
        ax21 = sns.scatterplot(ax=ax21, data=df_BO_valHTnmae_slf_b4[26:27], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=s_star)
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
        











    def plot_BO_individual_models(self, num_bvals, learning):
  
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


        df_BO = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b{num_bvals}_{learning}.csv')
        df_BO_valHTnmae = df_BO[["sum_nmae_valHT", "loss_valHT"]]

        pareto_df_BO_bool = paretoset(df_BO_valHTnmae, sense=["min", "min"])
        pareto_df_BO = df_BO_valHTnmae[pareto_df_BO_bool]

        df_BO_pareto = df_BO[pareto_df_BO_bool]
        pareto_filtered_df_BO = df_BO_pareto[(df_BO['Dt_nmae_valHT'] <= self.Dt_nmae) & 
                            (df_BO['Fp_nmae_valHT'] <= self.Fp_nmae) &
                            (df_BO['Dp_nmae_valHT'] <= self.Dp_nmae) &
                            (df_BO['sum_nmae_valHT'] <= self.sum_nmae)]
        

        if learning == 'slf':
            sorted_pareto_filtered_df_BO = pareto_filtered_df_BO.sort_values('loss_valHT')
        elif learning == 'sup':
            sorted_pareto_filtered_df_BO = pareto_filtered_df_BO.sort_values('sum_nmae_valHT')

        #sorted_pareto_filtered_df_BO.to_csv('test.csv', index=False)



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

        ax = sns.scatterplot(data=df_BO_valHTnmae, x='loss_valHT', y='sum_nmae_valHT', label='Experiment', c='tan')
        if learning == 'slf' and num_bvals != 11:
            ax = sns.scatterplot(data=pareto_df_BO, x='loss_valHT', y='sum_nmae_valHT', label='Pareto experiment', c='tomato')
        ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO, x='loss_valHT', y='sum_nmae_valHT', label='Passed pareto experiment', c='firebrick')
        if learning == 'slf':
            if num_bvals == 11:
                ax = sns.scatterplot(data=df_BO_valHTnmae[72:73], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=5000)
            else:
                ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO[:1], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=5000)
        elif learning == 'sup':
            ax = sns.scatterplot(data=sorted_pareto_filtered_df_BO[:1], x='loss_valHT', y='sum_nmae_valHT', label='Selected assed pareto experiment', c='dodgerblue', marker='*', s=5000)

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
        plt.savefig(os.path.join(self.dir_out, f'b{num_bvals}/BO_b{num_bvals}_{learning}.pdf'))    



    def plot_BO_all_models(self):

        df_BO_sup_b11 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b11_sup.csv')
        df_BO_sup_b5 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b5_sup.csv')
        df_BO_sup_b4 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b4_sup.csv')
        df_BO_slf_b11 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b11_slf.csv')
        df_BO_slf_b5 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b5_slf.csv')
        df_BO_slf_b4 = pd.read_csv(f'../simulations/simulations_analysis/df_trials_249_b4_slf.csv')

        df_BO_valHTnmae_sup_b11 = df_BO_sup_b11[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_sup_b5 = df_BO_sup_b5[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_sup_b4 = df_BO_sup_b4[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b11 = df_BO_slf_b11[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b5 = df_BO_slf_b5[["sum_nmae_valHT", "loss_valHT"]]
        df_BO_valHTnmae_slf_b4 = df_BO_slf_b4[["sum_nmae_valHT", "loss_valHT"]]


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

        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b11, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SL}^{11b}}$', c=[colors_cb[3]])
        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b5, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SL}^{5b}}$', c=[colors_cb[1]])
        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b4, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SL}^{4b}}$', c=[colors_cb[8]])
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b11, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SSL}^{11b}}$', c=[colors_cb[2]])
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b5, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SSL}^{5b}}$', c=[colors_cb[0]])
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b4, x='loss_valHT', y='sum_nmae_valHT', label=r'$\mathregular{DNN_{SSL}^{4b}}$', c=[colors_cb[4]])

        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b11[14:15], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[3]], s=2500)
        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b5[148:149], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[1]], s=2500)
        ax = sns.scatterplot(data=df_BO_valHTnmae_sup_b4[150:151], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[8]], s=2500)
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b11[72:73], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[2]], s=2500)
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b5[133:134], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[0]], s=2500)
        ax = sns.scatterplot(data=df_BO_valHTnmae_slf_b4[26:27], x='loss_valHT', y='sum_nmae_valHT', marker='*', c=[colors_b[4]], s=2500)


        ax.set(xlabel=r'signal-RMSE', ylabel=r'NMAE-SUM')

        ax.set_xlim(0.01, 0.062)
        ax.set_ylim(0.58, 1.9)

        lgnd = plt.legend()
        for i in range(6):
            lgnd.legendHandles[i]._sizes = [2500]

        fig.tight_layout()
        plt.savefig(os.path.join(self.dir_out, f'BO_all_models.pdf'))    