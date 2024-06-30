import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns
sns.set_theme()

import glob
import os
from tueplots import figsizes, fonts
from scipy.stats import wilcoxon


class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    def __init__(self, order=0, fformat='%.0f', offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat







class Patient_populationA_plot:
    def __init__(self, training_method):
        self.training_method = training_method
        self.dir_out = f'../dataA/analysis'
        if not os.path.exists(self.dir_out):
                os.makedirs(self.dir_out)

        self.df_param_stats = pd.read_csv(f'../dataA/analysis/df_param_stats_{training_method}.csv')
        self.df_param_diff_stats = pd.read_csv(f'../dataA/analysis/df_param_diff_stats_{training_method}.csv')
        self.df_CV_stats = pd.read_csv(f'../dataA/analysis/df_CV_stats_{training_method}.csv')


    def plot_ivim_params_stats(self):

        # Preparations
        sns.set_theme()
        sns.set_style('whitegrid')
        params = {'axes.labelsize': 25,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'legend.fontsize': 25,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        #boxprops={'alpha': 1}
        showfliers = True
        palette = ['steelblue', 'tomato']
        cut = 0

        # Plotting
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))
        [ax0, ax1, ax2, ax3] = axes


        sns.violinplot(ax=ax0, data=self.df_param_stats[self.df_param_stats.param == r'$D_t$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], inner=None, palette=palette, alpha=0.25, saturation=1, split=True, gap=0.05, cut=cut, legend=False)
        sns.violinplot(ax=ax1, data=self.df_param_stats[self.df_param_stats.param == r'$f_p$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], inner=None, palette=palette, alpha=0.25, saturation=1, split=True, gap=0.05, cut=cut, legend=False)
        sns.violinplot(ax=ax2, data=self.df_param_stats[self.df_param_stats.param == r'$D_p$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], inner=None, palette=palette, alpha=0.25, saturation=1, split=True, gap=0.05, cut=cut, legend=False)
        sns.violinplot(ax=ax3, data=self.df_param_stats[self.df_param_stats.param == r'signal-RMSE'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], inner=None, palette=palette, alpha=0.25, saturation=1, split=True, gap=0.05, cut=cut, legend=False)

        ax0 = sns.boxplot(ax=ax0, data=self.df_param_stats[self.df_param_stats.param == r'$D_t$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], palette=palette, showfliers=showfliers, saturation=1, width=0.25, gap=0.25)
        ax1 = sns.boxplot(ax=ax1, data=self.df_param_stats[self.df_param_stats.param == r'$f_p$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], palette=palette, showfliers=showfliers, saturation=1, width=0.25, gap=0.25)
        ax2 = sns.boxplot(ax=ax2, data=self.df_param_stats[self.df_param_stats.param == r'$D_p$'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], palette=palette, showfliers=showfliers, saturation=1, width=0.25, gap=0.25)
        ax3 = sns.boxplot(ax=ax3, data=self.df_param_stats[self.df_param_stats.param == r'signal-RMSE'], x='method', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], palette=palette, showfliers=showfliers, saturation=1, width=0.25, gap=0.25)


        for ax in axes:
            # https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot/72333641#72333641
            box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
            lines_per_boxplot = len(ax.lines) // len(box_patches)
            for i, patch in enumerate(box_patches):
                col = patch.get_facecolor()
                # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same color as above
                for i, line in enumerate(ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]):
                    if i == 5:  # fliers
                        line.set_color(col)
                        line.set_mfc(col)
                        line.set_mec('white')
                        line.set_marker('.')
                        line.set_markersize(10)


        ax0.set(xlabel=None, ylabel = r'Median $D_t$ [mm$^2$/s]')
        ax1.set(xlabel=None, ylabel = r'Median $f_p$')
        ax2.set(xlabel=None, ylabel = r'Median $D_p$ [mm$^2$/s]')
        ax3.set(xlabel=None, ylabel = r'Median signal-RMSE')

        ax0.yaxis.set_major_formatter(OOMFormatter(-3, "%.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-3, "%.0f"))

        ax0.legend().set_visible(False); ax1.legend().set_visible(False); ax2.legend().set_visible(False); ax3.legend().set_visible(False)

        handles, labels = ax3.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.075), ncol=2, fontsize=25)

        fig.tight_layout()
        plt.savefig(os.path.join(self.dir_out, f'patientA_population_params.pdf'), bbox_inches='tight')

        

    def plot_ivim_param_diffs_stats(self):

        # Preparations
        sns.set_theme()
        sns.set_style('whitegrid')
        params = {'axes.labelsize': 25,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'legend.fontsize': 25,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        showfliers = True
        palette = ['steelblue', 'coral', 'palevioletred', 'mediumseagreen']
        cut = 0

        # Plotting
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 15))
        [ax0, ax1, ax2] = axes


        sns.violinplot(ax=ax0, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $D_t$'], x='method', y='param_val', hue='method', inner=None, palette=palette, alpha=0.25, saturation=1, cut=cut, legend=False)
        sns.violinplot(ax=ax1, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $f_p$'], x='method', y='param_val', hue='method', inner=None, palette=palette, alpha=0.25, saturation=1, cut=cut, legend=False)
        sns.violinplot(ax=ax2, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $D_p$'], x='method', y='param_val', hue='method', inner=None, palette=palette, alpha=0.25, saturation=1, cut=cut, legend=False)

        ax0 = sns.boxplot(ax=ax0, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $D_t$'], x='method', y='param_val', hue='method', palette=palette, showfliers=showfliers, saturation=1, width=0.3, legend=False)
        ax1 = sns.boxplot(ax=ax1, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $f_p$'], x='method', y='param_val', hue='method', palette=palette, showfliers=showfliers, saturation=1, width=0.3, legend=False)
        ax2 = sns.boxplot(ax=ax2, data=self.df_param_diff_stats[self.df_param_diff_stats.param == r'MAE $D_p$'], x='method', y='param_val', hue='method', palette=palette, showfliers=showfliers, saturation=1, width=0.3, legend=False)


        for ax in axes:
            # https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot/72333641#72333641
            box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
            lines_per_boxplot = len(ax.lines) // len(box_patches)
            for i, patch in enumerate(box_patches):
                col = patch.get_facecolor()
                # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same color as above
                for j, line in enumerate(ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]):
                    if j == 5:  # fliers
                        line.set_color(col)
                        line.set_mfc(col)
                        line.set_mec('white')
                        line.set_marker('.')
                        line.set_markersize(12)

        ax0.set(xlabel=None, ylabel = r'Mean difference $D_t$ [mm$^2$/s]')
        ax1.set(xlabel=None, ylabel = r'Mean difference $f_p$')
        ax2.set(xlabel=None, ylabel = r'Mean difference $D_p$ [mm$^2$/s]')

        ax0.yaxis.set_major_formatter(OOMFormatter(-3, "%.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(-3, "%.0f"))

        fig.tight_layout()
        plt.savefig(os.path.join(self.dir_out, f'patientA_population_param_diffs.pdf'), bbox_inches='tight')



    def plot_CV_stats(self):

        # Preparations
        sns.set_theme()
        sns.set_style('whitegrid')
        params = {'axes.labelsize': 25,
                  'xtick.labelsize': 25,
                  'ytick.labelsize': 20,
                  'legend.fontsize': 25,
                  'legend.loc':'best',
                  'legend.framealpha': 0.75,
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        showfliers = True
        palette = ['steelblue', 'tomato']
        cut=0


        # Plotting
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

        sns.violinplot(ax=ax, data=self.df_CV_stats, x='param', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], inner=None, palette=palette, alpha=0.25, saturation=1, width=0.9, split=True, gap=0.05, cut=cut, legend=False)
        ax = sns.boxplot(ax=ax, data=self.df_CV_stats, x='param', y='param_val', hue='num_bvals', hue_order = ['5 b-values', '4 b-values'], palette=palette, showfliers=showfliers, saturation=1, width=0.25, gap=0.25)

        # https://stackoverflow.com/questions/36874697/how-to-edit-properties-of-whiskers-fliers-caps-etc-in-seaborn-boxplot/72333641#72333641
        box_patches = [patch for patch in ax.patches if type(patch) == matplotlib.patches.PathPatch]
        lines_per_boxplot = len(ax.lines) // len(box_patches)
        for i, patch in enumerate(box_patches):
            col = patch.get_facecolor()
                # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
                # Loop over them here, and use the same color as above
            for j, line in enumerate(ax.lines[i * lines_per_boxplot: (i + 1) * lines_per_boxplot]):
                if j == 5:  # fliers
                    line.set_color(col)
                    line.set_mfc(col)
                    line.set_mec('white')
                    line.set_marker('.')
                    line.set_markersize(12)

        ax.set(xlabel=None, ylabel = r'Mean CV')

        ax.legend().set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=25)


        fig.tight_layout()
        plt.savefig(os.path.join(self.dir_out, f'patientA_population_meanCV.pdf'), bbox_inches='tight')







    def print_ivim_params_stats(self, plot_name, num_bvals):
        if plot_name == 'params':
            df = self.df_param_stats
            description = 'IVIM parmaters and singal-RMSE'
            ivim_params = [r'$D_t$', r'$f_p$', r'$D_p$', 'signal-RMSE']
            factor = [1000, 1, 1000, 1]
        elif plot_name == 'param_diffs':
            df = self.df_param_diff_stats
            description = 'Parameter differences'
            ivim_params = [r'MAE $D_t$', r'MAE $f_p$', r'MAE $D_p$']
            factor = [1000, 1, 1000, 1]
        elif plot_name == 'CV':
            df = self.df_CV_stats
            description = 'CV'
            ivim_params = [r'$D_t$', r'$f_p$', r'$D_p$']
        else:
            print('Please enter a valid plot_name: params, param_diffs, or CV. ')


        print('*****************************************************')
        print(description)
        print('=====================================================')
        if plot_name == 'params':
            for i, param in enumerate(ivim_params):
                print(f'{param} x{1/factor[i]}')
                print('-----------')

                lsq_filtered = df[(df.num_bvals == num_bvals) & (df.param == param) & (df.method == 'LSQ')]['param_val'].to_numpy()*factor[i]
                seg_filtered = df[(df.num_bvals == num_bvals) & (df.param == param) & (df.method == 'SEG')]['param_val'].to_numpy()*factor[i]
                dnn_sup_filtered = df[(df.num_bvals == num_bvals) & (df.param == param) & (df.method == r'$\mathregular{DNN_{SL}}$')]['param_val'].to_numpy()*factor[i]
                dnn_slf_filtered = df[(df.num_bvals == num_bvals) & (df.param == param) & (df.method == r'$\mathregular{DNN_{SSL}}$')]['param_val'].to_numpy()*factor[i]

                q1_lsq, q3_lsq = np.percentile(lsq_filtered, [25 ,75])
                q1_seg, q3_seg = np.percentile(seg_filtered, [25 ,75])
                q1_sup, q3_sup = np.percentile(dnn_sup_filtered, [25 ,75])
                q1_slf, q3_slf = np.percentile(dnn_slf_filtered, [25 ,75])


                print(f'LSQ: \t\t Q2: {np.median(lsq_filtered):.2f},\t Q1: {q1_lsq:.2f} \t Q3: {q3_lsq:.2f}')
                print(f'SEG: \t\t Q2: {np.median(seg_filtered):.2f},\t Q1: {q1_seg:.2f} \t Q3: {q3_seg:.2f}')
                print(f'DNN_SL: \t Q2: {np.median(dnn_sup_filtered):.2f},\t Q1: {q1_sup:.2f} \t Q3: {q3_sup:.2f}')
                print(f'DNN_SSL: \t Q2: {np.median(dnn_slf_filtered):.2f},\t Q1: {q1_slf:.2f} \t Q3: {q3_slf:.2f}')
                print('=====================================================')
        elif plot_name == 'param_diffs':
            for i, param in enumerate(ivim_params):
                print(f'{param} x{1/factor[i]}')
                print('-----------')

                lsq_filtered = df[(df.param == param) & (df.method == 'LSQ')]['param_val'].to_numpy()*factor[i]
                seg_filtered = df[(df.param == param) & (df.method == 'SEG')]['param_val'].to_numpy()*factor[i]
                dnn_sup_filtered = df[(df.param == param) & (df.method == r'$\mathregular{DNN_{SL}}$')]['param_val'].to_numpy()*factor[i]
                dnn_slf_filtered = df[(df.param == param) & (df.method == r'$\mathregular{DNN_{SSL}}$')]['param_val'].to_numpy()*factor[i]
        
                q1_lsq, q3_lsq = np.percentile(lsq_filtered, [25 ,75])
                q1_seg, q3_seg = np.percentile(seg_filtered, [25 ,75])
                q1_sup, q3_sup = np.percentile(dnn_sup_filtered, [25 ,75])
                q1_slf, q3_slf = np.percentile(dnn_slf_filtered, [25 ,75])

                print(f'LSQ: \t\t Q2: {np.median(lsq_filtered):.2f},\t Q1: {q1_lsq:.2f} \t Q3: {q3_lsq:.2f}')
                print(f'SEG: \t\t Q2: {np.median(seg_filtered):.2f},\t Q1: {q1_seg:.2f} \t Q3: {q3_seg:.2f}')
                print(f'DNN_SL: \t Q2: {np.median(dnn_sup_filtered):.2f},\t Q1: {q1_sup:.2f} \t Q3: {q3_sup:.2f}')
                print(f'DNN_SSL: \t Q2: {np.median(dnn_slf_filtered):.2f},\t Q1: {q1_slf:.2f} \t Q3: {q3_slf:.2f}')

        else:
            for i, param in enumerate(ivim_params):
                filtered = df[(df.num_bvals == num_bvals) & (df.param == param)]['param_val'].to_numpy()
                q1, q3 = np.percentile(filtered, [25 ,75])

                print(f'{param}: \t\t Q2: {np.median(filtered):.2f},\t Q1: {q1:.2f} \t Q3: {q3:.2f}')



    def print_wilcoxon_signed_rank_test(self):
        df = self.df_param_stats
        ivim_params = [r'$D_t$', r'$f_p$', r'$D_p$', 'signal-RMSE']
        methods = ['LSQ', 'SEG', r'$\mathregular{DNN_{SL}}$', r'$\mathregular{DNN_{SSL}}$']
        factor = [1000, 1, 1000, 1]

        for i, param in enumerate(ivim_params):
            print('******************************************************************')
            print(param)
            print()
            for j, method in enumerate(methods):


                b5_filtered = df[(df.num_bvals == '5 b-values') & (df.param == param) & (df.method == method)]['param_val'].to_numpy()*factor[i]
                b4_filtered = df[(df.num_bvals == '4 b-values') & (df.param == param) & (df.method == method)]['param_val'].to_numpy()*factor[i]

                res = wilcoxon(b5_filtered, b4_filtered)
                print(f'{method}:\t\t {res}')
