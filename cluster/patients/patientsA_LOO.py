"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import glob
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import patients.patients as from_patients

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--patient_id', dest='patient_id', type=str)
parser.add_argument('--num_bvals', dest='num_bvals', type=int)
parser.add_argument('--training_method', dest='training_method', type=str)

args = parser.parse_args()
patient_id = args.patient_id
num_bvals = args.num_bvals
training_method = args.training_method



def run_algorithms_test(patient_id, mask_root_names, num_bvals, dataset_name, training_method, save_name = ''):
    # User specified variables
    cutoff = 200
    patient_ids = ['EMIN_1001', 'EMIN_1005', 'EMIN_1007', 'EMIN_1011', 'EMIN_1019', 
                   'EMIN_1020', 'EMIN_1022', 'EMIN_1032', 'EMIN_1038', 'EMIN_1042', 
                   'EMIN_1044', 'EMIN_1045', 'EMIN_1048', 'EMIN_1055', 'EMIN_1057', 
                   'EMIN_1060', 'EMIN_1064', 'EMIN_1066', 'EMIN_1068', 'EMIN_1075', 
                   'EMIN_1077', 'EMIN_1079', 'EMIN_1081', 'EMIN_1084', 'EMIN_1086', 
                   'EMIN_1090', 'EMIN_1092', 'EMIN_1093', 'EMIN_1096', 'EMIN_1097', 
                   'EMIN_1099']
    

    # remove patient id of interest from patient_ids. The removed patient id, specified by 'patient_id' will be
    # used for prediction, while the rest of the patient ids in 'patient_ids' will be used for training.
    patient_ids.remove(patient_id)

    for mask_root_name in mask_root_names:
        fnames_mask = glob.glob(f'../../../dataA/raw/{patient_id}/Segmentation/mask_{mask_root_name}' + '*.nii.gz')
        for fname_mask in fnames_mask: 
            mask_name = fname_mask.split(f'../../../dataA/raw/{patient_id}/Segmentation/mask_')[1].split('.nii.gz')[0]

            # lsq
            ivim_lsq = from_patients.IVIM_fit(patient_id, mask_name, method='lsq', 
                                              dataset_name=dataset_name,
                                              num_bvals=num_bvals,  
                                              save_name=save_name)
            ivim_lsq.fit_ivim()

            # seg
            ivim_seg = from_patients.IVIM_fit(patient_id, mask_name, method='seg', 
                                              dataset_name=dataset_name,
                                              num_bvals=num_bvals, 
                                              save_name=save_name)
            ivim_seg.fit_ivim(cutoff=cutoff)

    # self-supervised dnn
    ivim_dnn = from_patients.IVIM_dnn_LOO(patient_id, dataset_name, num_bvals, training_method, save_name=save_name)
    trained_model_ivim_net = ivim_dnn.train_ivim(patient_ids)
    ivim_dnn.predict_ivim(patient_id, trained_model_ivim_net)

    # repeat training for self-supervised dnn. Needed for consistency maps. Comment out if concistency maps are not of interest.
    for i in range(25):
        save_name_dnn = save_name + f'_r{i}'
        ivim_dnn = from_patients.IVIM_dnn_LOO(patient_id, dataset_name, num_bvals, training_method, save_name=save_name_dnn)
        trained_model_ivim_net = ivim_dnn.train_ivim(patient_ids)
        ivim_dnn.predict_ivim(patient_id, trained_model_ivim_net)


# User specified input! 
run_algorithms_test(patient_id, mask_root_names = ['GTVn'], num_bvals=num_bvals, dataset_name = 'dataA', training_method = training_method, save_name = f'patientsA_LOO_{training_method}_b{num_bvals}')

