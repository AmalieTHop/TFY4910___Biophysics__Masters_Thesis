
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import interpolate

import glob
import os
import time

import nibabel as nib

import algorithms.fitting_algos.LSQ_fitting as from_LSQ_fitting
import algorithms.fitting_algos.SEG_fitting as from_SEG_fitting
import algorithms.DNN.DNN as from_DNN
from patients.A_b5_hyperparams_selfsupervised import hyperparams_selfsupervised as A_b5_hyperparams_selfsupervised
from patients.A_b5_hyperparams_supervised import hyperparams_supervised as A_b5_hyperparams_supervised
from patients.A_b4_hyperparams_selfsupervised import hyperparams_selfsupervised as A_b4_hyperparams_selfsupervised
from patients.A_b4_hyperparams_supervised import hyperparams_supervised as A_b4_hyperparams_supervised


class IVIM_fit():
    def __init__(self, patient_id, mask_name, method, dataset_name, num_bvals, session_name = '', save_name='default'):
        self.patient_id = patient_id
        self.mask_name = mask_name
        self.method = method
        self.dataset_name = dataset_name
        self.num_bvals = num_bvals
        
        if dataset_name == 'dataA':
            if self.num_bvals == 5:
                self.bvals = np.array([0, 50, 100, 200, 800])
            elif self.num_bvals == 4:
                self.bvals = np.array([0, 50, 100, 800])
            else:
                print('Invalid number of b-values for dataA')
            self.dwi_4d_fname = glob.glob(f'../../../dataA/preprocessed/{patient_id}/dwi_4d/*.nii')[0]
            self.bvals_fname = f'../../../dataA/preprocessed/{patient_id}/bvals.npy'
            self.mask_fname = f'../../../dataA/preprocessed/{patient_id}/resampled_masks/{mask_name}/sorted_by_bval/resampled_mask_b0.nii'
            self.dir_out = f'../../../dataA/output/{save_name}/{patient_id}/{self.method}/{mask_name}'
        elif dataset_name == 'dataB':
            self.bvals = np.array([0, 50, 100, 800])
            self.dwi_4d_fname = glob.glob(f'../../../dataB/preprocessed/{patient_id}/{session_name}/dwi_4d/*.nii')[0]
            self.bvals_fname = f'../../../dataB/preprocessed/{patient_id}/{session_name}/bvals.npy'
            self.mask_fname = f'../../../dataB/segmentations/{patient_id}/{session_name}/{mask_name}_{session_name}.nii.gz'
            self.dir_out = f'../../../dataB/output/{save_name}/{patient_id}/{session_name}/{self.method}/{mask_name}'
        else:
            print('Invalid input for data_name')
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)

        self.dwi_4d_data = None
        self.dwi_4d_affine = None
        self.width = None
        self.height = None
        self.num_slices = None
        self.mask_data = None

        self.load_data()



    def load_data(self):
        dwi_4d_obj = nib.load(self.dwi_4d_fname)
        self.dwi_4d_data = dwi_4d_obj.get_fdata()
        self.dwi_4d_affine = dwi_4d_obj.affine
        self.width, self.height, self.num_slices, _ = self.dwi_4d_data.shape

        self.mask_data = nib.load(self.mask_fname).get_fdata()

        if self.dataset_name == 'dataA' and self.num_bvals == 4:
            self.dwi_4d_data = np.delete(self.dwi_4d_data, 3, 3)  # delete column with b=200


    def fit_ivim(self, bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3]), fitS0=True, cutoff=200):
        dwi_2d_data = np.reshape(self.dwi_4d_data, (self.width*self.height*self.num_slices, self.num_bvals))
        mask_1d_data = np.reshape(self.mask_data, (self.width*self.height*self.num_slices))

        np.savetxt('test', mask_1d_data)

        # fit
        params = ['Dt', 'fp', 'Dp', 'S0', 'rmse']
        if self.method == 'lsq':
            Dt_1d, Fp_1d, Dp_1d, S0_1d, rmse_1d = from_LSQ_fitting.fit_least_squares_array(self.bvals, dwi_2d_data, mask_1d_data, bounds=bounds, fitS0=fitS0)
        if self.method == 'seg':
            Dt_1d, Fp_1d, Dp_1d, S0_1d, rmse_1d = from_SEG_fitting.fit_least_squares_array(self.bvals, dwi_2d_data, mask_1d_data, bounds=bounds, fitS0=fitS0, cutoff=cutoff) #from_LSQ_fitting.fit_segmented_array(self.bvals, dwi_2d_data, mask_1d_data, bounds=bounds, cutoff=cutoff, p0=p0) 
        params_data = np.array([Dt_1d, Fp_1d, Dp_1d, S0_1d, rmse_1d])

        for i, param in enumerate(params):
            # save parameter maps
            param_3d = np.reshape(params_data[i], (self.width, self.height, self.num_slices))
            param_3d_lsq_nifti = nib.Nifti1Image(param_3d, self.dwi_4d_affine)
            nib.save(param_3d_lsq_nifti, os.path.join(self.dir_out, f'{param}.nii'))

        # calcualte median
        param_medians = np.median(params_data[:, mask_1d_data != 0], axis = 1)
        np.savetxt(os.path.join(self.dir_out, 'param_medians.txt'), param_medians)
        print(f'Successful calculation of parameter medians with {self.method}.')




class IVIM_dnn():
    def __init__(self, patient_id, mask_names, learning, dataset_name, num_bvals, training_method, session_name = '', save_name ='default'):
        self.patient_id = patient_id
        self.mask_names = mask_names
        self.learning = learning
        self.dataset_name = dataset_name
        self.num_bvals = num_bvals
        self.training_method = training_method
        self.session_name = session_name


        if dataset_name == 'dataA':
            if self.learning == 'slf':
                self.method = 'dnn_slf'
                self.dwi_4d_fname = glob.glob(f'../dataA/preprocessed/{patient_id}/dwi_4d/*.nii')[0]
                self.dir_out = f'../dataA/output/{save_name}/{patient_id}/{self.method}'
                if self.num_bvals == 5:
                    self.bvals = np.array([0, 50, 100, 200, 800])
                    arg = A_b5_hyperparams_selfsupervised()
                elif self.num_bvals == 4:
                    self.bvals = np.array([0, 50, 100, 800])
                    arg = A_b4_hyperparams_selfsupervised()
                else:
                    print('Invalid number of b-values for dataA')
            elif self.learning == 'sup':
                self.method = 'dnn_sup'
                self.dwi_4d_fname = glob.glob(f'../dataA/preprocessed/{patient_id}/dwi_4d/*.nii')[0]
                self.dir_out = f'../dataA/output/{save_name}/{patient_id}/{self.method}'
                if self.num_bvals == 5:
                    self.bvals = np.array([0, 50, 100, 200, 800])
                    arg = A_b5_hyperparams_supervised()
                elif self.num_bvals == 4:
                    self.bvals = np.array([0, 50, 100, 800])
                    arg = A_b4_hyperparams_supervised()
                else:
                    print('Invalid number of b-values for dataA')
            else:
                print('Enter valid learning method')
        else:
            print('Invalid input for data_name')

        self.arg = from_DNN.checkarg(arg)
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)

        
        # member variables below are given value though the functions load_data and preprocess_data, 
        # such that by the end of the construnctor all member varaibles has a defined value.
        self.dwi_4d_obj = None
        self.width = None
        self.height = None
        self.num_slices = None

        self.masks_data = None

        self.valid_idx_set_training = None
        self.valid_idx_set_prediction = None
        self.preprocessed_dwi_2d_training_data = None
        self.preprocessed_dwi_2d_prediction_data = None

        self.load_data()



    def load_data(self):
        self.dwi_4d_obj = nib.load(self.dwi_4d_fname)
        dwi_4d_data = self.dwi_4d_obj.get_fdata()
        self.width, self.height, self.num_slices, _ = dwi_4d_data.shape

        # load mask data
        self.masks_data = []
        for mask_name in self.mask_names:
            if self.dataset_name == 'dataA':
                if self.learning == 'slf':
                    mask_fname = f'../dataA/preprocessed/{self.patient_id}/resampled_masks/{mask_name}/sorted_by_bval/resampled_mask_b0.nii'
                elif self.learning == 'sup':
                    mask_fname = f'../dataA/preprocessed/{self.patient_id}/resampled_masks/{mask_name}/sorted_by_bval/resampled_mask_b0.nii'
            elif self.dataset_name == 'dataB':
                if self.learning == 'slf':
                    mask_fname = f'../dataB/segmentations/{self.patient_id}/{self.session_name}/{mask_name}_{self.session_name}.nii.gz'
                elif self.learning == 'sup':
                    mask_fname = f'../dataB/segmentations/{self.patient_id}/{self.session_name}/{mask_name}_{self.session_name}.nii.gz'

            mask_data = nib.load(mask_fname).get_fdata()

            self.masks_data.append(mask_data)
        
        # combine and reshape masked data
        combined_mask_data = np.sum(self.masks_data, axis=0)
        combined_mask_1d_data = np.reshape(combined_mask_data, self.width*self.height*self.num_slices).astype(bool)

        # reshape data to be trained
        if self.dataset_name == 'dataA' and self.num_bvals == 4:
            dwi_4d_data = np.delete(dwi_4d_data, 3, 3)  # delete column with b=200
        dwi_2d_data = np.reshape(dwi_4d_data, (self.width*self.height*self.num_slices, self.num_bvals))

        S0 = np.squeeze(dwi_2d_data[:, self.bvals == 0])
        # select training data
        if self.training_method == 'a':
            self.valid_idx_set_training = np.array(combined_mask_1d_data, dtype=bool)
        elif self.training_method == 'b':
            self.valid_idx_set_training = np.logical_or(S0 > (np.mean(S0)), combined_mask_1d_data)
        elif self.training_method == 'c':
            self.valid_idx_set_training = np.logical_and(S0 > (np.mean(S0)), ~combined_mask_1d_data)
        self.valid_idx_set_training[S0 == 0.0] = False
        valid_dwi_2d_training_data = dwi_2d_data[self.valid_idx_set_training, :]
        S0_training = S0[self.valid_idx_set_training]

        # select prediction data
        self.valid_idx_set_prediction = np.logical_or(S0 > (np.mean(S0)), combined_mask_1d_data)
        self.valid_idx_set_prediction[S0 == 0.0] = False
        valid_dwi_2d_prediction_data = dwi_2d_data[self.valid_idx_set_prediction, :]
        S0_prediction = S0[self.valid_idx_set_prediction]

        # normalise data
        self.preprocessed_dwi_2d_training_data = valid_dwi_2d_training_data / S0_training[:, None]
        self.preprocessed_dwi_2d_prediction_data = valid_dwi_2d_prediction_data / S0_prediction[:, None]


    def train_ivim(self):
        # ONLY for self-supervised!
        assert self.learning == 'slf'

        # train model
        start_time = time.time()
        net, loss_train, loss_valES, loss_valES_best = from_DNN.learn_selfsupervised(self.preprocessed_dwi_2d_training_data, self.bvals, self.arg)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for training: {elapsed_time}')

        # save 
        torch.save(net, os.path.join(self.dir_out, f'trained_model_net'))
        np.save(os.path.join(self.dir_out, f'loss_train'), loss_train)
        np.save(os.path.join(self.dir_out, f'loss_valES'), loss_valES)
        np.save(os.path.join(self.dir_out, f'loss_valES_best'), loss_valES_best)
        print("Trained model is saved")

        return net

    

    def predict_ivim(self, net):
        # define names IVIM params
        names = ['Dt', 'fp', 'Dp', 'S0', 'rmse']

        # predict
        start_time = time.time()
        params_pred, test_loss = from_DNN.predict_IVIM(self.preprocessed_dwi_2d_prediction_data, self.bvals, net, self.arg)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for training: {elapsed_time}')
        print(f'Test loss: {test_loss}')

        if self.arg.train_pars.use_cuda:
            torch.cuda.empty_cache()
        
        # save parameter maps and median parameter values
        param_medians = np.zeros((len(self.mask_names), len(names)))
        for j in range(len(names)):
            img = np.zeros([self.width * self.height * self.num_slices])
            img[self.valid_idx_set_prediction] = params_pred[j][0:sum(self.valid_idx_set_prediction)]
            img = np.reshape(img, [self.width, self.height, self.num_slices])
            nib.save(nib.Nifti1Image(img, self.dwi_4d_obj.affine, self.dwi_4d_obj.header), f'{self.dir_out}/{names[j]}.nii.gz')
            
            for k in range(len(self.mask_names)):
                param_medians[k][j] = np.median(img[self.masks_data[k]!= 0])

        for l in range(len(self.mask_names)):
            dir_out_pred = os.path.join(self.dir_out, f'{self.mask_names[l]}')
            if not os.path.exists(dir_out_pred):
                os.makedirs(dir_out_pred)
            np.savetxt(os.path.join(dir_out_pred, f'param_medians.txt'), param_medians[l])

        print(f'Successful calculation of parameter medians with {self.method}.')
    




class IVIM_dnn_LOO():
    def __init__(self, patient_id, dataset_name, num_bvals, training_method, session_name = '', save_name='default'):
        self.patient_id = patient_id
        self.dataset_name = dataset_name
        self.num_bvals = num_bvals
        self.training_method = training_method
        self.session_name = session_name
        self.method = 'dnn_slf'

        if self.dataset_name == 'dataA':
            if self.num_bvals == 5:
                self.bvals = np.array([0, 50, 100, 200, 800])
                arg = A_b5_hyperparams_selfsupervised()
            elif self.num_bvals == 4:
                self.bvals = np.array([0, 50, 100, 800])
                arg = A_b4_hyperparams_selfsupervised()
            else:
                print('Invalid number of b-values for dataA')
            self.dir_out = f'../../../dataA/output/{save_name}/{patient_id}/{self.method}'
            self.width = 364
            self.height = 256
        
        else:
            print('Invalid input for data_name')
        
        self.arg = from_DNN.checkarg(arg)
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)



    def preparations(self, patient_ids, mode):
        if mode == 'training':
            preprocessed_dwis_2d_data = []

        for patient_id in patient_ids:
            # load dwi data
            if self.dataset_name == 'dataA':
                dwi_4d_fname = glob.glob(f'../../../dataA/preprocessed/{patient_id}/dwi_4d/*.nii')[0]
            elif self.dataset_name == 'dataB':
                dwi_4d_fname = glob.glob(f'../../../dataB/preprocessed/{patient_id}/{self.session_name}/dwi_4d/*.nii')[0]
            else:
                print('Invalid input for data_name')
            dwi_4d_obj = nib.load(dwi_4d_fname)
            dwi_4d_data = dwi_4d_obj.get_fdata()
            num_slices = dwi_4d_data.shape[2]


            # load mask data
            mask_names_patient = np.load(f'../../../{self.dataset_name}/preprocessed/{patient_id}/mask_names.npy')

            mask_data_patient = []
            for mask_name in mask_names_patient:
                if self.dataset_name == 'dataA':
                    mask_fname = f'../../../dataA/preprocessed/{patient_id}/resampled_masks/{mask_name}/sorted_by_bval/resampled_mask_b0.nii'
                elif self.dataset_name == 'dataB':
                    mask_fname = f'../../../dataB/segmentations/{patient_id}/{self.session_name}/{mask_name}_{self.session_name}.nii.gz'
                    
                if mode == 'prediction':
                    dir_out_pred = os.path.join(self.dir_out, f'{mask_name}')
                    if not os.path.exists(dir_out_pred):
                        os.makedirs(dir_out_pred)
                    
                mask_data = nib.load(mask_fname).get_fdata()
                mask_data_patient.append(mask_data)

            ### combine and reshape mask
            combined_mask_data = np.sum(mask_data_patient, axis = 0)
            combined_mask_1d_data = np.reshape(combined_mask_data, self.width*self.height*num_slices).astype(bool)

            # reshape dwi data
            if self.dataset_name == 'dataA' and self.num_bvals == 4:
                dwi_4d_data = np.delete(dwi_4d_data, 3, 3)  # delete column with b=200
            dwi_2d_data = np.reshape(dwi_4d_data, (self.width*self.height*num_slices, self.num_bvals))

            # delete background and select valid data to train (and predict) on
            S0 = np.squeeze(dwi_2d_data[:, self.bvals == 0])
            if self.training_method == 'a': # only tumor
                valid_idx_set = np.array(combined_mask_1d_data, dtype=bool)
                print(self.training_method)
            elif self.training_method == 'b':
                valid_idx_set = np.logical_or(S0 > (np.mean(S0)), combined_mask_1d_data)
                print(self.training_method)
            else:
                print('Enter valid training method') 
            valid_idx_set[S0 == 0.0] = False
            valid_dwi_2d_data = dwi_2d_data[valid_idx_set, :]
            S0 = S0[valid_idx_set]    

            # normalise data
            preprocessed_dwi_2d_data = valid_dwi_2d_data / S0[:, None]
            
            ## add voxel to array of voxels from all patients 
            if mode == 'training':
                for voxl in preprocessed_dwi_2d_data:
                    preprocessed_dwis_2d_data.append(voxl)

            if mode == 'prediction':
                return preprocessed_dwi_2d_data, dwi_4d_obj, num_slices, mask_names_patient, mask_data_patient, valid_idx_set

        if mode == 'training':
            preprocessed_dwis_2d_data = np.array(preprocessed_dwis_2d_data)
            return preprocessed_dwis_2d_data
        else:
            print('Invalid input of mode.')



    def train_ivim(self, patient_ids):
        # preprocess data that will be used for training
        training_data = self.preparations(patient_ids,  mode='training')
        print(f'Shape: {training_data.shape}')

        # train model
        start_time = time.time()
        net, loss_train, loss_valES, loss_valES_best = from_DNN.learn_selfsupervised(training_data, self.bvals, self.arg)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for training: {elapsed_time}')

        # save 
        torch.save(net, os.path.join(self.dir_out, f'trained_model_net'))
        np.save(os.path.join(self.dir_out, f'loss_train'), loss_train)
        np.save(os.path.join(self.dir_out, f'loss_valES'), loss_valES)
        np.save(os.path.join(self.dir_out, f'loss_valES_best'), loss_valES_best)
        print("Trained model is saved")

        return net



    def predict_ivim(self, patient_id, net):
        # preprocess data that will be predicted
        preprocessed_dwi_2d_data, dwi_4d_obj, num_slices, mask_names_patient, mask_data_patient, valid_idx_set = self.preparations([patient_id], 'prediction')

        # define names IVIM params
        names = ['Dt', 'fp', 'Dp', 'S0', 'rmse']

        start_time = time.time()
        params_pred, test_loss = from_DNN.predict_IVIM(preprocessed_dwi_2d_data, self.bvals, net, self.arg)
        elapsed_time = time.time() - start_time
        print(f'Time elapsed for inference: {elapsed_time}')
        print(f'Test loss: {test_loss}')
            
        if self.arg.train_pars.use_cuda:
            torch.cuda.empty_cache()

        # save parameter maps and median parameter values
        param_medians = np.zeros((len(mask_names_patient), len(names)))
        for j in range(len(names)):
            img = np.zeros([self.width * self.height * num_slices])
            img[valid_idx_set] = params_pred[j][0:sum(valid_idx_set)]
            img = np.reshape(img, [self.width, self.height, num_slices])
            nib.save(nib.Nifti1Image(img, dwi_4d_obj.affine, dwi_4d_obj.header), f'{self.dir_out}/{names[j]}.nii.gz'),

            for k in range(len(mask_names_patient)):                    
                param_medians[k][j] = np.median(img[mask_data_patient[k]!= 0])
            
        for l in range(len(mask_names_patient)):
            dir_out_pred = os.path.join(self.dir_out, f'{mask_names_patient[l]}')
            if not os.path.exists(dir_out_pred):
                os.makedirs(dir_out_pred)
            np.savetxt(os.path.join(dir_out_pred, f'param_medians.txt'), param_medians[l])

        print(f'Successful calculation of parameter medians with {self.method}.')
    





