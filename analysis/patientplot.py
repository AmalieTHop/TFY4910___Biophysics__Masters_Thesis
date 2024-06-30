import os
import glob

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker
import seaborn as sns
from scipy import stats
sns.set_theme()

import nibabel as nib

import analysis.utils as from_utils
from tueplots import figsizes, fonts


class PatientpltIVIM:
    def __init__(self, patient_id, mask_name, average=0, save_name = ''):
        self.patient_id = patient_id
        self.mask_name = mask_name
        self.average = average
        self.save_name = save_name

        self.dir_in = f'../dataA/output/{save_name}/{patient_id}'
        self.dir_out = f'../dataA/plots/{patient_id}/{mask_name}/{save_name}'

        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)
        
        self.bvals = None
        self.z = None
        self.mask_data = None
        self.mask_slice = None
        self.mask_slice_dim_mask = None

        
        
        self.Dt_lsq_dim_mask, self.Fp_lsq_dim_mask, self.Dp_lsq_dim_mask = None, None, None
        self.Dt_seg_dim_mask, self.Fp_seg_dim_mask, self.Dp_seg_dim_mask = None, None, None
        self.Dt_dnn_sup_dim_mask, self.Fp_dnn_sup_dim_mask, self.Dp_dnn_sup_dim_mask = None, None, None
        self.Dt_dnn_slf_dim_mask, self.Fp_dnn_slf_dim_mask, self.Dp_dnn_slf_dim_mask = None, None, None

        self.shape_to_mask()

        
        
    def shape_to_mask(self):
        dwi_4d_fname = glob.glob(f'../dataA/preprocessed/{self.patient_id}/dwi_4d/*.nii')[0]
        bvals_fname = f'../dataA/preprocessed/{self.patient_id}/bvals.npy'
        mask_fname = f'../dataA/preprocessed/{self.patient_id}/resampled_masks/{self.mask_name}/sorted_by_bval/resampled_mask_b0.nii'

        dwi_4d_data = nib.load(dwi_4d_fname).get_fdata()
        self.bvals = np.load(bvals_fname)
        self.mask_data = nib.load(mask_fname).get_fdata()


        S0 = np.squeeze(dwi_4d_data[:, :, :, self.bvals == 0])   
        dwi_norm_4d_data = dwi_4d_data / S0[:, :, :, None]

        self.z = np.argmax(np.sum(self.mask_data, axis=(0,1)))
        mask_slice = self.mask_data[:, :, self.z]

        x_indecies, y_indecies = np.where(mask_slice == 1)
        x_min, x_max = np.min(x_indecies), np.max(x_indecies)
        y_min, y_max = np.min(y_indecies), np.max(y_indecies)


        Dt_lsq = nib.load(os.path.join(self.dir_in, f'lsq/{self.mask_name}/Dt.nii')).get_fdata()
        Fp_lsq = nib.load(os.path.join(self.dir_in, f'lsq/{self.mask_name}/fp.nii')).get_fdata()
        Dp_lsq = nib.load(os.path.join(self.dir_in, f'lsq/{self.mask_name}/Dp.nii')).get_fdata()
        S0_lsq = nib.load(os.path.join(self.dir_in, f'lsq/{self.mask_name}/S0.nii')).get_fdata()
        rmse_lsq = nib.load(os.path.join(self.dir_in, f'lsq/{self.mask_name}/rmse.nii')).get_fdata()

        Dt_seg = nib.load(os.path.join(self.dir_in, f'seg/{self.mask_name}/Dt.nii')).get_fdata()
        Fp_seg = nib.load(os.path.join(self.dir_in, f'seg/{self.mask_name}/fp.nii')).get_fdata() 
        Dp_seg = nib.load(os.path.join(self.dir_in, f'seg/{self.mask_name}/Dp.nii')).get_fdata()
        S0_seg = np.ones(np.shape(Dt_lsq))
        rmse_seg = nib.load(os.path.join(self.dir_in, f'seg/{self.mask_name}/rmse.nii')).get_fdata()

        Dt_dnn_sup = nib.load(os.path.join(self.dir_in, f'dnn_sup/Dt.nii.gz')).get_fdata()  ###
        Fp_dnn_sup = nib.load(os.path.join(self.dir_in, f'dnn_sup/fp.nii.gz')).get_fdata()  ###
        Dp_dnn_sup = nib.load(os.path.join(self.dir_in, f'dnn_sup/Dp.nii.gz')).get_fdata()  ###
        S0_dnn_sup = nib.load(os.path.join(self.dir_in, f'dnn_sup/S0.nii.gz')).get_fdata()  ###
        rmse_dnn_sup = nib.load(os.path.join(self.dir_in, f'dnn_sup/rmse.nii.gz')).get_fdata()  ###


        Dt_lsq_masked = np.multiply(Dt_lsq, self.mask_data)
        Fp_lsq_mased = np.multiply(Fp_lsq, self.mask_data)
        Dp_lsq_masked = np.multiply(Dp_lsq, self.mask_data)
        S0_lsq_masked = np.multiply(S0_lsq, self.mask_data)
        rmse_lsq_masked = np.multiply(rmse_lsq, self.mask_data)

        Dt_seg_masked = np.multiply(Dt_seg, self.mask_data)
        Fp_seg_masked = np.multiply(Fp_seg, self.mask_data)
        Dp_seg_masked = np.multiply(Dp_seg, self.mask_data)
        S0_seg_masked = np.multiply(S0_seg, self.mask_data)
        rmse_seg_masked = np.multiply(rmse_seg, self.mask_data)

        Dt_dnn_sup_masked = np.multiply(Dt_dnn_sup, self.mask_data)
        Fp_dnn_sup_masked = np.multiply(Fp_dnn_sup, self.mask_data)
        Dp_dnn_sup_masked = np.multiply(Dp_dnn_sup, self.mask_data)
        S0_dnn_sup_masked = np.multiply(S0_dnn_sup, self.mask_data)
        rmse_dnn_sup_masked =np.multiply(rmse_dnn_sup, self.mask_data)


        self.Dt_lsq_mask_vxls = Dt_lsq[(self.mask_data != 0)]
        self.Fp_lsq_mask_vxls = Fp_lsq[(self.mask_data != 0)]
        self.Dp_lsq_mask_vxls = Dp_lsq[(self.mask_data != 0)]
        self.rmse_lsq_mask_vxls = rmse_lsq[(self.mask_data != 0)]

        self.Dt_seg_mask_vxls = Dt_seg[(self.mask_data != 0)]
        self.Fp_seg_mask_vxls = Fp_seg[(self.mask_data != 0)]
        self.Dp_seg_mask_vxls = Dp_seg[(self.mask_data != 0)]
        self.rmse_seg_mask_vxls = rmse_seg[(self.mask_data != 0)]

        self.Dt_dnn_sup_mask_vxls = Dt_dnn_sup[(self.mask_data != 0)]
        self.Fp_dnn_sup_mask_vxls = Fp_dnn_sup[(self.mask_data != 0)]
        self.Dp_dnn_sup_mask_vxls = Dp_dnn_sup[(self.mask_data != 0)]
        self.rmse_dnn_sup_mask_vxls = rmse_dnn_sup[(self.mask_data != 0)]


        self.dwi_dim_mask = dwi_norm_4d_data[x_min:x_max, y_min:y_max, self.z]
        self.mask_slice_dim_mask = self.mask_data[x_min:x_max, y_min:y_max, self.z]

        self.Dt_lsq_dim_mask = Dt_lsq_masked[x_min:x_max, y_min:y_max, self.z]
        self.Fp_lsq_dim_mask = Fp_lsq_mased[x_min:x_max, y_min:y_max, self.z]
        self.Dp_lsq_dim_mask = Dp_lsq_masked[x_min:x_max, y_min:y_max, self.z]
        self.S0_lsq_dim_mask = S0_lsq_masked[x_min:x_max, y_min:y_max, self.z]
        self.rmse_lsq_dim_mask = rmse_lsq_masked[x_min:x_max, y_min:y_max, self.z]

        self.Dt_seg_dim_mask = Dt_seg_masked[x_min:x_max, y_min:y_max, self.z]
        self.Fp_seg_dim_mask = Fp_seg_masked[x_min:x_max, y_min:y_max, self.z]
        self.Dp_seg_dim_mask = Dp_seg_masked[x_min:x_max, y_min:y_max, self.z]
        self.S0_seg_dim_mask = S0_seg_masked[x_min:x_max, y_min:y_max, self.z]
        self.rmse_seg_dim_mask = rmse_seg_masked[x_min:x_max, y_min:y_max, self.z]

        self.Dt_dnn_sup_dim_mask = Dt_dnn_sup_masked[x_min:x_max, y_min:y_max, self.z]
        self.Fp_dnn_sup_dim_mask = Fp_dnn_sup_masked[x_min:x_max, y_min:y_max, self.z]
        self.Dp_dnn_sup_dim_mask = Dp_dnn_sup_masked[x_min:x_max, y_min:y_max, self.z]
        self.S0_dnn_sup_dim_mask = S0_dnn_sup_masked[x_min:x_max, y_min:y_max, self.z]
        self.rmse_dnn_sup_dim_mask = rmse_dnn_sup_masked[x_min:x_max, y_min:y_max, self.z]


        if self.average:
            self.Dt_dnn_slf_mask_vxls = np.zeros((self.average, int(np.sum(self.mask_data))))
            self.Fp_dnn_slf_mask_vxls = np.zeros((self.average, int(np.sum(self.mask_data))))
            self.Dp_dnn_slf_mask_vxls = np.zeros((self.average, int(np.sum(self.mask_data))))
            self.rmse_dnn_slf_mask_vxls = np.zeros((self.average, int(np.sum(self.mask_data))))

            self.Dt_dnn_slf_dim_mask = np.zeros((self.average, int(x_max - x_min), int(y_max - y_min)))
            self.Fp_dnn_slf_dim_mask = np.zeros((self.average, int(x_max - x_min), int(y_max - y_min)))
            self.Dp_dnn_slf_dim_mask = np.zeros((self.average, int(x_max - x_min), int(y_max - y_min)))
            self.S0_dnn_slf_dim_mask = np.zeros((self.average, int(x_max - x_min), int(y_max - y_min)))
            self.rmse_dnn_slf_dim_mask = np.zeros((self.average, int(x_max - x_min), int(y_max - y_min)))

            for i in range(self.average):
                Dt_dnn_slf = nib.load(f'../dataA/output/{self.save_name}_r{i}/{self.patient_id}/dnn_slf/Dt.nii.gz').get_fdata()
                Fp_dnn_slf = nib.load(f'../dataA/output/{self.save_name}_r{i}/{self.patient_id}/dnn_slf/fp.nii.gz').get_fdata()
                Dp_dnn_slf = nib.load(f'../dataA/output/{self.save_name}_r{i}/{self.patient_id}/dnn_slf/Dp.nii.gz').get_fdata()
                S0_dnn_slf = nib.load(f'../dataA/output/{self.save_name}_r{i}/{self.patient_id}/dnn_slf/S0.nii.gz').get_fdata()
                rmse_dnn_slf = nib.load(f'../dataA/output/{self.save_name}_r{i}/{self.patient_id}/dnn_slf/rmse.nii.gz').get_fdata()

                Dt_dnn_slf_masked = np.multiply(Dt_dnn_slf, self.mask_data)
                Fp_dnn_slf_masked = np.multiply(Fp_dnn_slf, self.mask_data)
                Dp_dnn_slf_masked = np.multiply(Dp_dnn_slf, self.mask_data)
                S0_dnn_slf_masked = np.multiply(S0_dnn_slf, self.mask_data)
                rmse_dnn_slf_masked =np.multiply(rmse_dnn_slf, self.mask_data)

                self.Dt_dnn_slf_mask_vxls[i] = Dt_dnn_slf[(self.mask_data != 0)]
                self.Fp_dnn_slf_mask_vxls[i] = Fp_dnn_slf[(self.mask_data != 0)]
                self.Dp_dnn_slf_mask_vxls[i] = Dp_dnn_slf[(self.mask_data != 0)]
                self.rmse_dnn_slf_mask_vxls[i] = rmse_dnn_slf[(self.mask_data != 0)]

                self.Dt_dnn_slf_dim_mask[i] = Dt_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
                self.Fp_dnn_slf_dim_mask[i] = Fp_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
                self.Dp_dnn_slf_dim_mask[i] = Dp_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
                self.S0_dnn_slf_dim_mask[i] = S0_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
                self.rmse_dnn_slf_dim_mask[i] = rmse_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]

                
        else:
            Dt_dnn_slf = nib.load(os.path.join(self.dir_in, f'dnn_slf/Dt.nii.gz')).get_fdata()
            Fp_dnn_slf = nib.load(os.path.join(self.dir_in, f'dnn_slf/fp.nii.gz')).get_fdata()
            Dp_dnn_slf = nib.load(os.path.join(self.dir_in, f'dnn_slf/Dp.nii.gz')).get_fdata()
            S0_dnn_slf = nib.load(os.path.join(self.dir_in, f'dnn_slf/S0.nii.gz')).get_fdata()
            rmse_dnn_slf = nib.load(os.path.join(self.dir_in, f'dnn_slf/rmse.nii.gz')).get_fdata()

            Dt_dnn_slf_masked = np.multiply(Dt_dnn_slf, self.mask_data)
            Fp_dnn_slf_masked = np.multiply(Fp_dnn_slf, self.mask_data)
            Dp_dnn_slf_masked = np.multiply(Dp_dnn_slf, self.mask_data)
            S0_dnn_slf_masked = np.multiply(S0_dnn_slf, self.mask_data)
            rmse_dnn_slf_masked =np.multiply(rmse_dnn_slf, self.mask_data)

            self.Dt_dnn_slf_mask_vxls = Dt_dnn_slf[(self.mask_data != 0)]
            self.Fp_dnn_slf_mask_vxls = Fp_dnn_slf[(self.mask_data != 0)]
            self.Dp_dnn_slf_mask_vxls = Dp_dnn_slf[(self.mask_data != 0)]
            self.rmse_dnn_slf_mask_vxls = rmse_dnn_slf[(self.mask_data != 0)]

            self.Dt_dnn_slf_dim_mask = Dt_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
            self.Fp_dnn_slf_dim_mask = Fp_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
            self.Dp_dnn_slf_dim_mask = Dp_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
            self.S0_dnn_slf_dim_mask = S0_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]
            self.rmse_dnn_slf_dim_mask = rmse_dnn_slf_masked[x_min:x_max, y_min:y_max, self.z]


    

    def plt_masked_param_map(self):
        # Preprocessing
        if self.average:
            Dt_dnn_slf_dim_mask = np.mean(self.Dt_dnn_slf_dim_mask, axis=0)
            Fp_dnn_slf_dim_mask = np.mean(self.Fp_dnn_slf_dim_mask, axis=0)
            Dp_dnn_slf_dim_mask = np.mean(self.Dp_dnn_slf_dim_mask, axis=0)
            rmse_dnn_slf_dim_mask = np.mean(self.rmse_dnn_slf_dim_mask, axis=0)
        else:
            Dt_dnn_slf_dim_mask = self.Dt_dnn_slf_dim_mask
            Fp_dnn_slf_dim_mask = self.Fp_dnn_slf_dim_mask
            Dp_dnn_slf_dim_mask = self.Dp_dnn_slf_dim_mask
            rmse_dnn_slf_dim_mask = self.rmse_dnn_slf_dim_mask

        # Preparations
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        alpha = self.mask_slice_dim_mask
        cbformat_Dt = from_utils.OOMFormatter(-3, "%.1f")
        cbformat_Dp = from_utils.OOMFormatter(-3, "%.0f")


        #Plotting
        fig, axes = plt.subplots(nrows = 4, ncols = 4, figsize=(15, 10), constrained_layout = True)
        [(ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23), (ax30, ax31, ax32, ax33)] = axes
        
        
        extend_max = 'max'
        extend_min = 'min'
        im00 = ax00.imshow(self.Dt_lsq_dim_mask, alpha=alpha, vmin=0, vmax=0.002)
        im01 = ax01.imshow(self.Dt_seg_dim_mask, alpha=alpha, vmin=0, vmax=0.002)
        im02 = ax02.imshow(self.Dt_dnn_sup_dim_mask, alpha=alpha, vmin=0, vmax=0.002)
        im03 = ax03.imshow(Dt_dnn_slf_dim_mask, alpha=alpha, vmin=0, vmax=0.002)

        im10 = ax10.imshow(self.Fp_lsq_dim_mask, alpha=alpha, vmin=0, vmax = 0.35)
        im11 = ax11.imshow(self.Fp_seg_dim_mask, alpha=alpha, vmin=0, vmax = 0.35)
        im12 = ax12.imshow(self.Fp_dnn_sup_dim_mask, alpha=alpha, vmin=0, vmax = 0.35)
        im13 = ax13.imshow(Fp_dnn_slf_dim_mask, alpha=alpha, vmin=0, vmax = 0.35)

        im20 = ax20.imshow(self.Dp_lsq_dim_mask, alpha=alpha, vmin=0, vmax=0.070)
        im21 = ax21.imshow(self.Dp_seg_dim_mask, alpha=alpha, vmin=0, vmax=0.070)
        im22 = ax22.imshow(self.Dp_dnn_sup_dim_mask, alpha=alpha, vmin=0, vmax=0.070)
        im23 = ax23.imshow(Dp_dnn_slf_dim_mask, alpha=alpha, vmin=0, vmax=0.070)

        im30 = ax30.imshow(self.rmse_lsq_dim_mask, alpha=alpha, vmin=0, vmax=0.05)
        im31 = ax31.imshow(self.rmse_seg_dim_mask, alpha=alpha, vmin=0, vmax=0.05)
        im32 = ax32.imshow(self.rmse_dnn_sup_dim_mask, alpha=alpha, vmin=0, vmax=0.05)
        im33 = ax33.imshow(rmse_dnn_slf_dim_mask, alpha=alpha, vmin=0, vmax=0.05)

        column_titles = ['LSQ', 'SEG', r'$\mathregular{DNN_{SL}}$', r'$\mathregular{DNN_{SSL}}$']
        pad = 25
        for ax, column_title in zip(axes[0], column_titles):
            ax.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', fontsize=32)
        
        cbar00 = fig.colorbar(im03, ax=axes[0, :], format=cbformat_Dt, extend=extend_max, aspect=12.5, pad=0.025)
        cbar00.set_label(label=r'$D_t$ [mm$^2$/s]', size=25)
        cbar00.ax.tick_params(labelsize=17.5)

        cbar10 = fig.colorbar(im13, ax=axes[1, :], extend=extend_max, aspect=12.5, pad=0.052)
        cbar10.set_label(label=r'$f_p$', size=25)
        cbar10.ax.tick_params(labelsize=17.5)

        cbar20 = fig.colorbar(im23, ax=axes[2, :], format=cbformat_Dp, extend=extend_max, aspect=12.5, pad=0.025)
        cbar20.set_label(label=r'$D_p$ [mm$^2$/s]', size=25)
        cbar20.ax.tick_params(labelsize=17.5)

        cbar30 = fig.colorbar(im33, ax=axes[3, :], extend=extend_max, aspect=12.5, pad=0.052)
        cbar30.set_label(label=r'signal-RMSE', size=25)
        cbar30.ax.tick_params(labelsize=17.5)


        for ax in axes.reshape(-1):
            ax.axis("off")

        if self.average:
            plt.savefig(os.path.join(self.dir_out, f'param_map_{self.patient_id}_avg'), bbox_inches='tight')   
            plt.savefig(os.path.join(self.dir_out, f'param_map_{self.patient_id}_avg.svg'), bbox_inches='tight')      
        else:
            plt.savefig(os.path.join(self.dir_out, f'param_map_{self.patient_id}'), bbox_inches='tight')   
            plt.savefig(os.path.join(self.dir_out, f'param_map_{self.patient_id}.svg'), bbox_inches='tight')  



    def plt_CV(self):

        # Preprocessing
        CV_Dt_dnn_slf_dim_mask = np.std(self.Dt_dnn_slf_dim_mask, axis=0) / np.mean(self.Dt_dnn_slf_dim_mask, axis=0)
        CV_Fp_dnn_slf_dim_mask = np.std(self.Fp_dnn_slf_dim_mask, axis=0) / np.mean(self.Fp_dnn_slf_dim_mask, axis=0)
        CV_Dp_dnn_slf_dim_mask = np.std(self.Dp_dnn_slf_dim_mask, axis=0) / np.mean(self.Dp_dnn_slf_dim_mask, axis=0)


        # Preparations
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'plasma',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        alpha = self.mask_slice_dim_mask
        cv_Dt_max = 0.035
        cv_Fp_max = 0.35
        cv_Dp_max = 0.15
        cv_min = 0
        extend_max = 'max'

        #Plotting
        fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize=(15, 10), constrained_layout = True)
        [ax0, ax1, ax2] = axes
        
        im0 = ax0.imshow(CV_Dt_dnn_slf_dim_mask, alpha=alpha, vmin = cv_min, vmax=cv_Dt_max)
        im1 = ax1.imshow(CV_Fp_dnn_slf_dim_mask, alpha=alpha, vmin = cv_min, vmax=cv_Fp_max)
        im2 = ax2.imshow(CV_Dp_dnn_slf_dim_mask, alpha=alpha, vmin = cv_min, vmax=cv_Dp_max)

        column_title = 'Dummy'
        pad = 25
        ax0.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                     xycoords='axes fraction', textcoords='offset points',
                     ha='center', va='baseline', color='white', fontsize=35)
        
        cbar00 = fig.colorbar(im0, ax=ax0, extend=extend_max, aspect=12.5)
        cbar00.set_label(label=r'CV $D_t$', size=30)
        cbar00.ax.tick_params(labelsize=20)

        cbar10 = fig.colorbar(im1, ax=ax1, extend=extend_max, aspect=12.5)
        cbar10.set_label(label=r'CV $f_p$', size=30)
        cbar10.ax.tick_params(labelsize=20)

        cbar20 = fig.colorbar(im2, ax=ax2, extend=extend_max, aspect=12.5)
        cbar20.set_label(label=r'CV $D_p$', size=30)
        cbar20.ax.tick_params(labelsize=20)

        for ax in axes.reshape(-1):
            ax.axis("off")

        plt.savefig(os.path.join(self.dir_out, f'CV_map_{self.patient_id}'), bbox_inches='tight')   
        plt.savefig(os.path.join(self.dir_out, f'CV_map_{self.patient_id}.svg'), bbox_inches='tight')  



    def plt_consistency(self):
        # Preparations
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        alpha = self.mask_slice_dim_mask
        cbformat_Dt = from_utils.OOMFormatter(-3, "%.1f")
        cbformat_Dp = from_utils.OOMFormatter(-3, "%.0f")
        extend_max = 'max'

        #Plotting
        fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize=(15, 10), constrained_layout = True)
        [(ax00, ax01, ax02), (ax10, ax11, ax12), (ax20, ax21, ax22)] = axes

        
        im00 = ax00.imshow(self.Dt_dnn_slf_dim_mask[0], alpha=alpha, vmin=0, vmax=0.002)
        im01 = ax01.imshow(self.Dt_dnn_slf_dim_mask[1], alpha=alpha, vmin=0, vmax=0.002)
        im02 = ax02.imshow(self.Dt_dnn_slf_dim_mask[2], alpha=alpha, vmin=0, vmax=0.002)

        im10 = ax10.imshow(self.Fp_dnn_slf_dim_mask[0], alpha=alpha, vmin=0, vmax = 0.35)
        im11 = ax11.imshow(self.Fp_dnn_slf_dim_mask[1], alpha=alpha, vmin=0, vmax = 0.35)
        im12 = ax12.imshow(self.Fp_dnn_slf_dim_mask[2], alpha=alpha, vmin=0, vmax = 0.35)

        im20 = ax20.imshow(self.Dp_dnn_slf_dim_mask[0], alpha=alpha, vmin=0, vmax=0.070)
        im21 = ax21.imshow(self.Dp_dnn_slf_dim_mask[1], alpha=alpha, vmin=0, vmax=0.070)
        im22 = ax22.imshow(self.Dp_dnn_slf_dim_mask[2], alpha=alpha, vmin=0, vmax=0.070)

        
        column_titles = ['Instance 1', 'Instance 2', 'Instance 3']
        pad = 25
        for ax, column_title in zip(axes[0], column_titles):
            ax.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', fontsize=35)
        
        cbar00 = fig.colorbar(im02, ax=axes[0, :], format=cbformat_Dt, extend=extend_max, aspect=12.5, pad=0.025)
        cbar00.set_label(label=r'$D_t$ [mm$^2$/s]', size=30)
        cbar00.ax.tick_params(labelsize=20)

        cbar10 = fig.colorbar(im12, ax=axes[1, :], extend=extend_max, aspect=12.5, pad=0.048)
        cbar10.set_label(label=r'$f_p$', size=30)
        cbar10.ax.tick_params(labelsize=20)

        cbar20 = fig.colorbar(im22, ax=axes[2, :], format=cbformat_Dp, extend=extend_max, aspect=12.5, pad=0.025)
        cbar20.set_label(label=r'$D_p$ [mm$^2$/s]', size=30)
        cbar20.ax.tick_params(labelsize=20)

        for ax in axes.reshape(-1):
            ax.axis("off")

        plt.savefig(os.path.join(self.dir_out, f'consistency_param_maps_{self.patient_id}'), bbox_inches='tight') 
        plt.savefig(os.path.join(self.dir_out, f'consistency_param_maps_{self.patient_id}.svg'), bbox_inches='tight')  







    def plt_whole_param_map(self, learning):
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'turbo',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)


        params = ['Dt', 'fp', 'Dp', 'rmse']
        param_labels = [r'$D_t$ [mm$^2$/s]', r'$f_p$ [%]', r'$D_p$ [mm$^2$/s]', r'RMSE']
        bounds_min = [0, 0, 0, 0]
        bounds_max = [0.002, 35, 0.075, 0.05]

        levels = np.array([0.5])

        extend_max = 'max'
        cbformat = from_utils.OOMFormatter(-3, "%.0f")

        for i in range(len(params)):
            param_data = nib.load(os.path.join(self.dir_in, f'dnn_{learning}/{params[i]}.nii.gz')).get_fdata()

            cbformat = from_utils.OOMFormatter(-3, "%.0f")
            if params[i] == 'fp':
                cbformat = from_utils.OOMFormatter(0, "%.0f")
                param_data = 100*param_data

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(param_data[:, :, self.z].T, vmin=bounds_min[i], vmax=bounds_max[i])
            cbar = fig.colorbar(im, ax = ax, format=cbformat, extend=extend_max)
            cbar.set_label(label=param_labels[i])
            c = ax.contour(self.mask_data[:, :, self.z].T, levels, colors='white')

            fig.tight_layout()
            plt.savefig(os.path.join(self.dir_out, f'param_map_{params[i]}_{learning}'))      


    def plt_param_distr(self):
            sns.set_theme()
            params = {'axes.labelsize': 20,
                    'axes.titlesize': 22.5,
                    'lines.marker': 'none',
                    'xtick.labelsize': 15,
                    'ytick.labelsize': 15,
                    'legend.fontsize': 17.5,
                    'legend.loc':'upper right',
                    'legend.framealpha': 0.75,
                    'savefig.format': 'pdf'
                    }
            plt.rcParams.update(params)


            alpha = 0.25
            num_bins = 50
            Dt_bins = np.linspace(0, 0.003, num_bins)
            Fp_bins = np.linspace(0, 50, num_bins)
            Dp_bins = np.linspace(0.005, 0.1, num_bins)
            rmse_bins = np.linspace(0, 0.08, num_bins)

            kde_Dt_range = np.linspace(0, 0.003, 500)
            kde_Fp_range = np.linspace(0, 50, 500)
            kde_Dp_range = np.linspace(0.005, 0.1, 500)
            kde_rmse_range = np.linspace(0, 0.08, 500)

            kde_Dt_lsq = stats.gaussian_kde(self.Dt_lsq_mask_vxls)
            kde_Dt_seg = stats.gaussian_kde(self.Dt_seg_mask_vxls)
            kde_Dt_dnn_sup = stats.gaussian_kde(self.Dt_dnn_sup_mask_vxls)
            kde_Dt_dnn_slf = stats.gaussian_kde(self.Dt_dnn_slf_mask_vxls)
            kde_Fp_lsq = stats.gaussian_kde(self.Fp_lsq_mask_vxls*100)
            kde_Fp_seg = stats.gaussian_kde(self.Fp_seg_mask_vxls*100)
            kde_Fp_dnn_sup = stats.gaussian_kde(self.Fp_dnn_sup_mask_vxls*100)
            kde_Fp_dnn_slf = stats.gaussian_kde(self.Fp_dnn_slf_mask_vxls*100)
            kde_Dp_lsq = stats.gaussian_kde(self.Dp_lsq_mask_vxls)
            kde_Dp_seg = stats.gaussian_kde(self.Dp_seg_mask_vxls)
            kde_Dp_dnn_sup = stats.gaussian_kde(self.Dp_dnn_sup_mask_vxls)
            kde_Dp_dnn_slf = stats.gaussian_kde(self.Dp_dnn_slf_mask_vxls)
            kde_rmse_lsq = stats.gaussian_kde(self.rmse_lsq_mask_vxls)
            kde_rmse_seg = stats.gaussian_kde(self.rmse_seg_mask_vxls)
            kde_rmse_dnn_sup = stats.gaussian_kde(self.rmse_dnn_sup_mask_vxls)
            kde_rmse_dnn_slf = stats.gaussian_kde(self.rmse_dnn_slf_mask_vxls)


            fig, [(ax00), (ax10), (ax20), (ax30)] = plt.subplots(nrows = 4, ncols = 1, figsize=(15, 20))
            #fig.suptitle(suptitle, fontsize=20)

            ax00.hist(self.Dt_lsq_mask_vxls, density=True, bins=Dt_bins, color='C0', alpha=alpha, label='LSQ')
            ax00.hist(self.Dt_seg_mask_vxls, density=True, bins=Dt_bins, color='C1', alpha=alpha, label='SEG')
            ax00.hist(self.Dt_dnn_sup_mask_vxls, density=True, bins=Dt_bins, color='C4', alpha=alpha, label=r'$DNN_{sup}$')
            ax00.hist(self.Dt_dnn_slf_mask_vxls, density=True, bins=Dt_bins, color='C2', alpha=alpha, label=r'$DNN_{slf}$')
            ax00.plot(kde_Dt_range, kde_Dt_lsq.pdf(kde_Dt_range), color='C0', label=r'$PDF_{LSQ}$')
            ax00.plot(kde_Dt_range, kde_Dt_seg.pdf(kde_Dt_range), color='C1', label=r'$PDF_{SEG}$')
            ax00.plot(kde_Dt_range, kde_Dt_dnn_sup.pdf(kde_Dt_range), color='C4', label=r'$PDF_{DNN_{slf}}$')
            ax00.plot(kde_Dt_range, kde_Dt_dnn_slf.pdf(kde_Dt_range), color='C2', label=r'$PDF_{DNN_{sup}}$')
            ax00.set(xlim=(0, 0.003), xlabel=r'$D_t$ [mm$^2$/s]', ylabel='Probability density')
            ax00.legend()

            ax10.hist(self.Fp_lsq_mask_vxls*100, density=True, bins=Fp_bins, color='C0', alpha=alpha, label='LSQ')
            ax10.hist(self.Fp_seg_mask_vxls*100, density=True, bins=Fp_bins, color='C1', alpha=alpha, label='SEG')
            ax10.hist(self.Fp_dnn_sup_mask_vxls*100, density=True, bins=Fp_bins, color='C4', alpha=alpha, label=r'$DNN_{sup}$')
            ax10.hist(self.Fp_dnn_slf_mask_vxls*100, density=True, bins=Fp_bins, color='C2', alpha=alpha, label=r'$DNN_{slf}$')
            ax10.plot(kde_Fp_range, kde_Fp_lsq.pdf(kde_Fp_range), color='C0', label=r'$PDF_{LSQ}$')
            ax10.plot(kde_Fp_range, kde_Fp_seg.pdf(kde_Fp_range),color='C1',  label=r'$PDF_{SEG}$')
            ax00.plot(kde_Fp_range, kde_Fp_dnn_sup.pdf(kde_Fp_range), color='C4', label=r'$PDF_{DNN_{slf}}$')
            ax00.plot(kde_Fp_range, kde_Fp_dnn_slf.pdf(kde_Fp_range), color='C2', label=r'$PDF_{DNN_{sup}}$')
            ax10.set(xlim=(0, 50), xlabel=r'$f_p$ [%]', ylabel='Probability density')
            ax10.legend()

            ax20.hist(self.Dp_lsq_mask_vxls, density=True, bins=Dp_bins, color='C0', alpha=alpha, label='LSQ')
            ax20.hist(self.Dp_seg_mask_vxls, density=True, bins=Dp_bins, color='C1', alpha=alpha, label='SEG')
            ax20.hist(self.Dp_dnn_sup_mask_vxls, density=True, bins=Dp_bins, color='C4', alpha=alpha, label=r'$DNN_{sup}$')
            ax20.hist(self.Dp_dnn_slf_mask_vxls, density=True, bins=Dp_bins, color='C2', alpha=alpha, label=r'$DNN_{slf}$')
            ax20.plot(kde_Dp_range, kde_Dp_lsq.pdf(kde_Dp_range), color='C0', label=r'$PDF_{LSQ}$')
            ax20.plot(kde_Dp_range, kde_Dp_seg.pdf(kde_Dp_range), color='C1', label=r'$PDF_{SEG}$')
            ax20.plot(kde_Dp_range, kde_Dp_dnn_sup.pdf(kde_Dp_range), color='C4', label=r'$PDF_{DNN_{slf}}$')
            ax20.plot(kde_Dp_range, kde_Dp_dnn_slf.pdf(kde_Dp_range), color='C2', label=r'$PDF_{DNN_{sup}}$')
            ax20.set(xlim=(0, 0.1), xlabel=r'$D_p$ [mm$^2$/s]', ylabel='Probability density')
            ax20.legend()

            ax30.hist(self.rmse_lsq_mask_vxls, density=True, bins=rmse_bins, color='C0', alpha=alpha, label='LSQ')
            ax30.hist(self.rmse_seg_mask_vxls, density=True, bins=rmse_bins, color='C1', alpha=alpha, label='SEG')
            ax30.hist(self.rmse_dnn_sup_mask_vxls, density=True, bins=rmse_bins, color='C4', alpha=alpha, label=r'$DNN_{sup}$')
            ax30.hist(self.rmse_dnn_slf_mask_vxls, density=True, bins=rmse_bins, color='C2', alpha=alpha, label=r'$DNN_{slf}$')
            ax30.plot(kde_rmse_range, kde_rmse_lsq.pdf(kde_rmse_range), color='C0', label=r'$PDF_{LSQ}$')
            ax30.plot(kde_rmse_range, kde_rmse_seg.pdf(kde_rmse_range), color='C1', label=r'$PDF_{SEG}$')
            ax30.plot(kde_rmse_range, kde_rmse_dnn_sup.pdf(kde_rmse_range), color='C4', label=r'$PDF_{DNN_{sup}}$')
            ax30.plot(kde_rmse_range, kde_rmse_dnn_slf.pdf(kde_rmse_range), color='C2', label=r'$PDF_{DNN_{slf}}$')
            ax30.set(xlim=(0, 0.08), xlabel=r'RMSE', ylabel='Probability density')
            ax30.legend()


            fig.tight_layout()
            plt.savefig(os.path.join(self.dir_out, f'param_dists_{self.patient_id}'))


    def plt_signal_curves(self):
        x_indecies, y_indecies = np.where((self.rmse_lsq_dim_mask <= 0.010) & (self.mask_slice_dim_mask != 0) & (self.Dp_lsq_dim_mask >= 0.09))
        print(x_indecies.shape)
        x_indecies, y_indecies = x_indecies[:10], y_indecies[:10]

        sns.set_theme()
        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'legend.fontsize': 17.5,
                  'legend.loc':'upper right',
                  'legend.framealpha': 0.75,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(params)
        save_name = f'IVIM_signal_curves_{self.patient_id}'
        plot_b_vals = np.linspace(0, 800, 801)


        for i, (x, y) in enumerate(zip(x_indecies, y_indecies)):
            Sb_lsq = from_utils.ivim(plot_b_vals, self.Dt_lsq_dim_mask[x, y], self.Fp_lsq_dim_mask[x, y], self.Dp_lsq_dim_mask[x, y], self.S0_lsq_dim_mask[x, y])
            Sb_seg = from_utils.ivim(plot_b_vals, self.Dt_seg_dim_mask[x, y], self.Fp_seg_dim_mask[x, y], self.Dp_seg_dim_mask[x, y], self.S0_seg_dim_mask[x, y])
            Sb_dnn_sup = from_utils.ivim(plot_b_vals, self.Dt_dnn_sup_dim_mask[x, y], self.Fp_dnn_sup_dim_mask[x, y], self.Dp_dnn_sup_dim_mask[x, y], self.S0_dnn_sup_dim_mask[x, y])
            Sb_dnn_slf = from_utils.ivim(plot_b_vals, self.Dt_dnn_slf_dim_mask[x, y], self.Fp_dnn_slf_dim_mask[x, y], self.Dp_dnn_slf_dim_mask[x, y], self.S0_dnn_slf_dim_mask[x, y])

            print(f'LSQ:\t\t {np.around(self.Dt_lsq_dim_mask[x, y]*1000, 2)}\t{np.around(self.Fp_lsq_dim_mask[x, y]*100, 2)}\t{np.around(self.Dp_lsq_dim_mask[x, y]*1000, 2)}\t{np.around(self.rmse_lsq_dim_mask[x, y], 3)}')
            print(f'SEG:\t\t {np.around(self.Dt_seg_dim_mask[x, y]*1000, 2)}\t{np.around(self.Fp_seg_dim_mask[x, y]*100, 2)}\t{np.around(self.Dp_seg_dim_mask[x, y]*1000, 2)}\t{np.around(self.rmse_seg_dim_mask[x, y], 3)}')
            print(f'DNN_sup:\t {np.around(self.Dt_dnn_sup_dim_mask[x, y]*1000, 2)}\t{np.around(self.Fp_dnn_sup_dim_mask[x, y]*100, 2)}\t{np.around(self.Dp_dnn_sup_dim_mask[x, y]*1000, 2)}\t{np.around(self.rmse_dnn_sup_dim_mask[x, y], 3)}')
            print(f'DNN_slf:\t {np.around(self.Dt_dnn_slf_dim_mask[x, y]*1000, 2)}\t{np.around(self.Fp_dnn_slf_dim_mask[x, y]*100, 2)}\t{np.around(self.Dp_dnn_slf_dim_mask[x, y]*1000, 2)}\t{np.around(self.rmse_dnn_slf_dim_mask[x, y], 3)}')
            print()

            plt.figure(figsize = [10,5])
            plt.plot(plot_b_vals, Sb_lsq, label = 'LSQ', color='C0')
            plt.plot(plot_b_vals, Sb_seg, label = 'SEG', color='C1')
            plt.plot(plot_b_vals, Sb_dnn_sup, label = r'DNN_{sup}', color='C4')
            plt.plot(plot_b_vals, Sb_dnn_slf, label = r'$DNN_{slf}$', color='C2')
            plt.scatter(self.bvals, self.dwi_dim_mask[x, y], label = "Measured data", color='C8')
            plt.xlabel(r'$b$-values')
            plt.ylabel(r'$S_{norm}(b)$')
            plt.yticks()
            plt.xticks()
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir_out, save_name + f'_{i}'))





class Patientplt_across_bvals:
    def __init__(self, patient_id, mask_name, average, training_method, patientplt_b5, patientplt_b4):
        self.patient_id = patient_id
        self.average = average
        self.training_method = training_method
        self.pp_b5 = patientplt_b5
        self.pp_b4 = patientplt_b4
  
        self.dir_out = f'../dataA/plots/{patient_id}/{mask_name}'

        self.pps = [self.pp_b5, self.pp_b4]
        self.nums_bvals = [5, 4]



    def plt_diff_param_maps(self):

        # Precprocessing #
        mask_slice_dim_mask = self.pp_b5.mask_slice_dim_mask

        Dt_lsq_dim_mask_diff = np.subtract(self.pp_b5.Dt_lsq_dim_mask, self.pp_b4.Dt_lsq_dim_mask)
        Dt_seg_dim_mask_diff = np.subtract(self.pp_b5.Dt_seg_dim_mask, self.pp_b4.Dt_seg_dim_mask)
        Dt_dnn_sup_dim_mask_diff = np.subtract(self.pp_b5.Dt_dnn_sup_dim_mask, self.pp_b4.Dt_dnn_sup_dim_mask)
        Dt_dnn_slf_dim_mask_diff = np.subtract(self.pp_b5.Dt_dnn_slf_dim_mask, self.pp_b4.Dt_dnn_slf_dim_mask)

        Fp_lsq_dim_mask_diff = np.subtract(self.pp_b5.Fp_lsq_dim_mask, self.pp_b4.Fp_lsq_dim_mask)
        Fp_seg_dim_mask_diff = np.subtract(self.pp_b5.Fp_seg_dim_mask, self.pp_b4.Fp_seg_dim_mask)
        Fp_dnn_sup_dim_mask_diff = np.subtract(self.pp_b5.Fp_dnn_sup_dim_mask, self.pp_b4.Fp_dnn_sup_dim_mask)
        Fp_dnn_slf_dim_mask_diff = np.subtract(self.pp_b5.Fp_dnn_slf_dim_mask, self.pp_b4.Fp_dnn_slf_dim_mask)

        Dp_lsq_dim_mask_diff = np.subtract(self.pp_b5.Dp_lsq_dim_mask, self.pp_b4.Dp_lsq_dim_mask)
        Dp_seg_dim_mask_diff = np.subtract(self.pp_b5.Dp_seg_dim_mask, self.pp_b4.Dp_seg_dim_mask)
        Dp_dnn_sup_dim_mask_diff = np.subtract(self.pp_b5.Dp_dnn_sup_dim_mask, self.pp_b4.Dp_dnn_sup_dim_mask)
        Dp_dnn_slf_dim_mask_diff = np.subtract(self.pp_b5.Dp_dnn_slf_dim_mask, self.pp_b4.Dp_dnn_slf_dim_mask)

        if self.average:
            Dt_dnn_slf_dim_mask_diff = np.mean(Dt_dnn_slf_dim_mask_diff, axis=0)
            Fp_dnn_slf_dim_mask_diff = np.mean(Fp_dnn_slf_dim_mask_diff, axis=0)
            Dp_dnn_slf_dim_mask_diff = np.mean(Dp_dnn_slf_dim_mask_diff, axis=0)


        # Preparations
        v_Dt = 0.0002
        v_Fp = 0.1
        v_Dp = 0.04

        params = {'axes.labelsize': 20,
                  'axes.titlesize': 22.5,
                  'axes.grid': False,
                  'image.cmap': 'RdBu_r',
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'savefig.format': 'pdf'
                  }
        plt.rcParams.update(fonts.neurips2021())
        plt.rcParams.update(params)

        alpha = mask_slice_dim_mask
        cbformat_Dt = from_utils.OOMFormatter(-3, "%.1f")
        cbformat_Dp = from_utils.OOMFormatter(-3, "%.0f")
        extend_max = 'both'

        # Plotting
        fig, axes = plt.subplots(nrows = 3, ncols = 4, figsize=(17, 9), constrained_layout = True)
        [(ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13), (ax20, ax21, ax22, ax23)] = axes
        
        im00 = ax00.imshow(Dt_lsq_dim_mask_diff, alpha=alpha, vmin=-v_Dt, vmax=v_Dt)
        im01 = ax01.imshow(Dt_seg_dim_mask_diff, alpha=alpha, vmin=-v_Dt, vmax=v_Dt)
        im02 = ax02.imshow(Dt_dnn_sup_dim_mask_diff, alpha=alpha, vmin=-v_Dt, vmax=v_Dt)
        im03 = ax03.imshow(Dt_dnn_slf_dim_mask_diff, alpha=alpha, vmin=-v_Dt, vmax=v_Dt)

        im10 = ax10.imshow(Fp_lsq_dim_mask_diff, alpha=alpha, vmin=-v_Fp, vmax=v_Fp)
        im11 = ax11.imshow(Fp_seg_dim_mask_diff, alpha=alpha, vmin=-v_Fp, vmax=v_Fp)
        im12 = ax12.imshow(Fp_dnn_sup_dim_mask_diff, alpha=alpha, vmin=-v_Fp, vmax=v_Fp)
        im13 = ax13.imshow(Fp_dnn_slf_dim_mask_diff, alpha=alpha, vmin=-v_Fp, vmax=v_Fp)

        im20 = ax20.imshow(Dp_lsq_dim_mask_diff, alpha=alpha, vmin=-v_Dp, vmax=v_Dp)
        im21 = ax21.imshow(Dp_seg_dim_mask_diff, alpha=alpha, vmin=-v_Dp, vmax=v_Dp)
        im22 = ax22.imshow(Dp_dnn_sup_dim_mask_diff, alpha=alpha, vmin=-v_Dp, vmax=v_Dp)
        im23 = ax23.imshow(Dp_dnn_slf_dim_mask_diff, alpha=alpha, vmin=-v_Dp, vmax=v_Dp)

        column_titles = ['LSQ', 'SEG', r'$\mathregular{DNN_{SL}}$', r'$\mathregular{DNN_{SSL}}$']
        pad = 25
        for ax, column_title in zip(axes[0], column_titles):
            ax.annotate(column_title, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        ha='center', va='baseline', fontsize=32)
        
        cbar00 = fig.colorbar(im03, ax=axes[0, :], format=cbformat_Dt, extend=extend_max, aspect=12.5, pad=0.025)
        cbar00.set_label(label=r'$D_{t}^{\mathregular{b5}} - D_{t}^{\mathregular{b4}}$ [mm$^2$/s]', size=25)
        cbar00.ax.tick_params(labelsize=17.5)

        cbar10 = fig.colorbar(im13, ax=axes[1, :],extend=extend_max, aspect=12.5, pad=0.045)
        cbar10.set_label(label=r'$f_{p}^{\mathregular{b5}} - f_{p}^{\mathregular{b4}}$', size=25)
        cbar10.ax.tick_params(labelsize=17.5)

        cbar20 = fig.colorbar(im23, ax=axes[2, :], format=cbformat_Dp, extend=extend_max, aspect=12.5, pad=0.025)
        cbar20.set_label(label=r'$D_{p}^{\mathregular{b5}} - D_{p}^{\mathregular{b4}}$ [mm$^2$/s]', size=25)
        cbar20.ax.tick_params(labelsize=17.5)

        for ax in axes.reshape(-1):
            ax.axis("off")

        plt.savefig(os.path.join(self.dir_out, f'diff_maps_{self.training_method}.pdf'), bbox_inches='tight')



    def add_to_df_param_stats(self, df_param_stats):

        for num_bvals, pp in zip(self.nums_bvals, self.pps):
            lsq_Dt = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'LSQ', 'param': r'$D_t$', 'param_val': [np.median(pp.Dt_lsq_mask_vxls)]})
            seg_Dt = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'SEG', 'param': r'$D_t$', 'param_val': [np.median(pp.Dt_seg_mask_vxls)]})
            dnn_sup_Dt = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'$D_t$', 'param_val': [np.median(pp.Dt_dnn_sup_mask_vxls)]})
            dnn_slf_Dt = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'$D_t$', 'param_val': [np.median(pp.Dt_dnn_slf_mask_vxls)]})

            lsq_Fp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'LSQ', 'param': r'$f_p$', 'param_val': [np.median(pp.Fp_lsq_mask_vxls)]})
            seg_Fp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'SEG', 'param': r'$f_p$', 'param_val': [np.median(pp.Fp_seg_mask_vxls)]})
            dnn_sup_Fp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'$f_p$', 'param_val': [np.median(pp.Fp_dnn_sup_mask_vxls)]})
            dnn_slf_Fp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'$f_p$', 'param_val': [np.median(pp.Fp_dnn_slf_mask_vxls)]})

            lsq_Dp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'LSQ', 'param': r'$D_p$', 'param_val': [np.median(pp.Dp_lsq_mask_vxls)]})
            seg_Dp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'SEG', 'param': r'$D_p$', 'param_val': [np.median(pp.Dp_seg_mask_vxls)]})
            dnn_sup_Dp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'$D_p$', 'param_val': [np.median(pp.Dp_dnn_sup_mask_vxls)]})
            dnn_slf_Dp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'$D_p$', 'param_val': [np.median(pp.Dp_dnn_slf_mask_vxls)]})

            lsq_rmse = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'LSQ', 'param': 'signal-RMSE', 'param_val': [np.median(pp.rmse_lsq_mask_vxls)]})
            seg_rmse = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': f'SEG', 'param': 'signal-RMSE', 'param_val': [np.median(pp.rmse_seg_mask_vxls)]})
            dnn_sup_rmse = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SL}}$', 'param': 'signal-RMSE', 'param_val': [np.median(pp.rmse_dnn_sup_mask_vxls)]})
            dnn_slf_rmse = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'method': r'$\mathregular{DNN_{SSL}}$', 'param': 'signal-RMSE', 'param_val': [np.median(pp.rmse_dnn_slf_mask_vxls)]})
        
            df_param_stats = pd.concat([df_param_stats,
                                       lsq_Dt, seg_Dt, dnn_sup_Dt, dnn_slf_Dt,
                                       lsq_Fp, seg_Fp, dnn_sup_Fp, dnn_slf_Fp,
                                       lsq_Dp, seg_Dp, dnn_sup_Dp, dnn_slf_Dp, 
                                       lsq_rmse, seg_rmse, dnn_sup_rmse, dnn_slf_rmse])

        df_param_stats.to_csv(f'../dataA/analysis/df_param_stats_{self.training_method}.csv', index=False)
        return df_param_stats
        

    def add_to_df_param_diffs_stats(self, df_param_diffs_stats):

        Dt_lsq_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dt_lsq_mask_vxls, self.pp_b4.Dt_lsq_mask_vxls)))
        Dt_seg_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dt_seg_mask_vxls, self.pp_b4.Dt_seg_mask_vxls)))
        Dt_dnn_sup_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dt_dnn_sup_mask_vxls, self.pp_b4.Dt_dnn_sup_mask_vxls)))
        Dt_dnn_slf_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dt_dnn_slf_mask_vxls, self.pp_b4.Dt_dnn_slf_mask_vxls)))

        Fp_lsq_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Fp_lsq_mask_vxls, self.pp_b4.Fp_lsq_mask_vxls)))
        Fp_seg_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Fp_seg_mask_vxls, self.pp_b4.Fp_seg_mask_vxls)))
        Fp_dnn_sup_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Fp_dnn_sup_mask_vxls, self.pp_b4.Fp_dnn_sup_mask_vxls)))
        Fp_dnn_slf_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Fp_dnn_slf_mask_vxls, self.pp_b4.Fp_dnn_slf_mask_vxls)))

        Dp_lsq_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dp_lsq_mask_vxls, self.pp_b4.Dp_lsq_mask_vxls)))
        Dp_seg_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dp_seg_mask_vxls, self.pp_b4.Dp_seg_mask_vxls)))
        Dp_dnn_sup_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dp_dnn_sup_mask_vxls, self.pp_b4.Dp_dnn_sup_mask_vxls)))
        Dp_dnn_slf_mask_vxls_diff = np.mean(np.absolute(np.subtract(self.pp_b5.Dp_dnn_slf_mask_vxls, self.pp_b4.Dp_dnn_slf_mask_vxls)))

        mae_lsq_Dt = pd.DataFrame({'patient_id': self.patient_id, 'method': f'LSQ', 'param': r'MAE $D_t$', 'param_val': [Dt_lsq_mask_vxls_diff]})
        mae_seg_Dt = pd.DataFrame({'patient_id': self.patient_id, 'method': f'SEG', 'param': r'MAE $D_t$', 'param_val': [Dt_seg_mask_vxls_diff]})
        mae_dnn_sup_Dt = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'MAE $D_t$', 'param_val': [Dt_dnn_sup_mask_vxls_diff]})
        mae_dnn_slf_Dt = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'MAE $D_t$', 'param_val': [Dt_dnn_slf_mask_vxls_diff]})

        mae_lsq_Fp = pd.DataFrame({'patient_id': self.patient_id, 'method': f'LSQ', 'param': r'MAE $f_p$', 'param_val': [Fp_lsq_mask_vxls_diff]})
        mae_seg_Fp = pd.DataFrame({'patient_id': self.patient_id, 'method': f'SEG', 'param': r'MAE $f_p$', 'param_val': [Fp_seg_mask_vxls_diff]})
        mae_dnn_sup_Fp = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'MAE $f_p$', 'param_val': [Fp_dnn_sup_mask_vxls_diff]})
        mae_dnn_slf_Fp = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'MAE $f_p$', 'param_val': [Fp_dnn_slf_mask_vxls_diff]})

        mae_lsq_Dp = pd.DataFrame({'patient_id': self.patient_id, 'method': f'LSQ', 'param': r'MAE $D_p$', 'param_val': [Dp_lsq_mask_vxls_diff]})
        mae_seg_Dp = pd.DataFrame({'patient_id': self.patient_id, 'method': f'SEG', 'param': r'MAE $D_p$', 'param_val': [Dp_seg_mask_vxls_diff]})
        mae_dnn_sup_Dp = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SL}}$', 'param': r'MAE $D_p$', 'param_val': [Dp_dnn_sup_mask_vxls_diff]})
        mae_dnn_slf_Dp = pd.DataFrame({'patient_id': self.patient_id, 'method': r'$\mathregular{DNN_{SSL}}$', 'param': r'MAE $D_p$', 'param_val': [Dp_dnn_slf_mask_vxls_diff]})

        df_param_diff_stats = pd.concat([df_param_diff_stats,
                                       mae_lsq_Dt, mae_seg_Dt, mae_dnn_sup_Dt, mae_dnn_slf_Dt,
                                       mae_lsq_Fp, mae_seg_Fp, mae_dnn_sup_Fp, mae_dnn_slf_Fp,
                                       mae_lsq_Dp, mae_seg_Dp, mae_dnn_sup_Dp, mae_dnn_slf_Dp])

        df_param_diffs_stats.to_csv(f'../dataA/analysis/df_param_diffs_stats_{self.training_method}.csv', index=False)
        return df_param_diffs_stats   




    def add_to_df_CV_stats(self, df_CV_stats):

        for num_bvals, pp in zip(self.nums_bvals, self.pps):

            CV_Dt_dnn_slf_mask_vxls = np.std(pp.Dt_dnn_slf_mask_vxls, axis=0) / np.mean(pp.Dt_dnn_slf_mask_vxls, axis=0)
            CV_Fp_dnn_slf_mask_vxls = np.std(pp.Fp_dnn_slf_mask_vxls, axis=0) / np.mean(pp.Fp_dnn_slf_mask_vxls, axis=0)
            CV_Dp_dnn_slf_mask_vxls = np.std(pp.Dp_dnn_slf_mask_vxls, axis=0) / np.mean(pp.Dp_dnn_slf_mask_vxls, axis=0)

            CV_dnn_slf_Dt = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'param': r'$D_t$', 'param_val': [np.nanmedian(CV_Dt_dnn_slf_mask_vxls)]})
            CV_dnn_slf_Fp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'param': r'$f_p$', 'param_val': [np.nanmedian(CV_Fp_dnn_slf_mask_vxls)]})
            CV_dnn_slf_Dp = pd.DataFrame({'patient_id': self.patient_id, 'num_bvals': f'{num_bvals} b-values', 'param': r'$D_p$', 'param_val': [np.nanmedian(CV_Dp_dnn_slf_mask_vxls)]})
        
            df_CV_stats = pd.concat([df_CV_stats,
                                     CV_dnn_slf_Dt, CV_dnn_slf_Fp, CV_dnn_slf_Dp])

        df_CV_stats.to_csv(f'../dataA/analysis/df_CV_stats_{self.training_method}.csv', index=False)
        return df_CV_stats
        

