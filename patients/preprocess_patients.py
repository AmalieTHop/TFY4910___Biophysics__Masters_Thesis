"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import numpy as np
import glob
import os

import pydicom
import dicom2nifti
import SimpleITK as sitk



class Preprocess:
    def __init__(self, patient_id, dir_preprocessed_patient, dir_raw_dwi):
        self.patient_id = patient_id
        self.dir_raw_dwi = dir_raw_dwi
        self.fnames_raw_dwi = dir_raw_dwi + '/*.IMA'

        self.dir_preprocessed_patient = dir_preprocessed_patient
        if not os.path.exists(self.dir_preprocessed_patient):
            os.makedirs(self.dir_preprocessed_patient)

        self.dir_sorted_by_slice_num = os.path.join(self.dir_preprocessed_patient, 'sorted_by_slice_num')
        self.dir_sorted_by_bval = os.path.join(self.dir_preprocessed_patient, 'sorted_by_bval')

        self.num_slices = None
        self.bvals = None

        print("--------------------------------------------------")
        print(f"Preprocessing data for patient {self.patient_id}:")



    def sort_files_by_slice_number_and_bval(self):
        """
        # Makes the folders sorted_by_slice_num and sorted_by_bval. sorted_by_slice_num has subfolders, 
        # where each subfolder corrresponds to a slice and contains the dicom files with the different 
        # b-values for the given slice. sorted_by_bval has subfolders, where each subfolder corresponds to
        # a b-value and contains dicom fiels with the different slices for the given b-value.  
        """

        unsorted_files = []
    
        # open all files in source, extract Slice Location and B-value and store the data of 
        # the files as a subarray in the array unsorted_files
        paths = glob.glob(self.fnames_raw_dwi)
        for path in paths:
            dataset = pydicom.dcmread(path)
            
            slice_location = round(dataset.get("SliceLocation","NA"), ndigits = 3)
            bval = dataset[0x0019, 0x0100c].value
            
            unsorted_files.append([path, dataset, slice_location, bval])
        
        # sorted lists of unique slice locations and b-values
        sorted_set_sl = sorted(set(el[2] for el in unsorted_files))
        sorted_set_b = sorted(set(el[3] for el in unsorted_files))

        # assgin value to member variables
        self.num_slices = len(sorted_set_sl)
        self.bvals = sorted_set_b

        # sort the files on slice location
        sorted_files_sl = sorted(unsorted_files, key = lambda unsorted_file: unsorted_file[2])
        # sort the files first on b-value, then on slice location
        sorted_files_b = sorted(unsorted_files, key = lambda unsorted_file: (unsorted_file[3], unsorted_file[2]))

        # save files in sorted file system based on slice number
        for i in range(len(sorted_files_sl)):
            path, dataset, slice_location, bval = sorted_files_sl[i]
            slice_number = sorted_set_sl.index(slice_location)
            fileName = f'sn{slice_number}_b{str(bval)}.dcm'
        
            # save files to a nested folder structure
            if not os.path.exists(os.path.join(self.dir_sorted_by_slice_num, f'sn{slice_number}')):
                os.makedirs(os.path.join(self.dir_sorted_by_slice_num, f'sn{slice_number}'))
            dataset.save_as(os.path.join(self.dir_sorted_by_slice_num, f'sn{slice_number}', fileName))
        print("\tSuccessful sorting of files by slice number.")

        # save files in sorted file system based on b-value
        for i in range(len(sorted_files_b)):
            path, dataset, slice_location, bval = sorted_files_b[i]
            slice_number = sorted_set_sl.index(slice_location) #+ 1
            fileName = f'sn{slice_number}_b{str(bval)}.dcm'

            # save files to a nested folder structure
            if not os.path.exists(os.path.join(self.dir_sorted_by_bval, f'b{bval}')):
                os.makedirs(os.path.join(self.dir_sorted_by_bval, f'b{bval}'))
            dataset.save_as(os.path.join(self.dir_sorted_by_bval, f'b{bval}', fileName))
        print("\tSuccessful sorting of files by b-value.")
        


    def create_bvalsfile(self):
        """
        Writes the b-values to file.
        """

        np.save(os.path.join(self.dir_preprocessed_patient + '/bvals'), self.bvals) 
        print("\tSuccessful writing of b-values to file.")



    def create_dwi_4d_as_nifti(self):
        """
        Create a nifti file with four dimentions, where the first three dimensions corresponded to 
        the spatial dimensions x, y, and z, and the fourth to the b-value.
        """

        dir_dwi_4d = os.path.join(self.dir_preprocessed_patient, 'dwi_4d')
        if not os.path.exists(dir_dwi_4d):
            os.makedirs(dir_dwi_4d)

        dicom2nifti.convert_directory(self.dir_raw_dwi, dir_dwi_4d, compression=False, reorient=False)
        print("\tSuccessful creating of 4d dwi data as nifti file.")


    
    def bval_series_as_nifti(self):
        """
        # Converts each b-value dicom image series (e.g. all the slices), made by the function
        # sort_files_by_slice_number_and_bval, into a nifti file.
        """
        
        for bval in self.bvals:
            src_dir = os.path.join(self.dir_sorted_by_bval, f'b{bval}')
            dst_fname = os.path.join(self.dir_sorted_by_bval, f'b{bval}/b{bval}')

            dicom2nifti.dicom_series_to_nifti(src_dir, dst_fname, reorient_nifti=False)
        
        print("\tSuccessful coverting of series of dcm files into nifti files for each b-value.")
    


    def resample_mask(self, mask_name, fname_mask):
        """
        Resamples the mask with name mask_name. 
        """

        dir_resampled_mask = os.path.join(self.dir_preprocessed_patient, f'resampled_masks/{mask_name}/sorted_by_bval')
        if not os.path.exists(dir_resampled_mask):
            os.makedirs(dir_resampled_mask)

        # scale mask by factor 10
        raw_mask_img = sitk.ReadImage(fname_mask) * 10  # scaling as the matrix only takes integer
    

        for bval in self.bvals:
            src_path_to_ref_img = os.path.join(self.dir_preprocessed_patient, f'sorted_by_bval/b{bval}/b{bval}.nii')
            dst_path_to_resampled_mask = os.path.join(dir_resampled_mask, f'resampled_mask_b{bval}.nii')

            # reference image
            ref_img = sitk.ReadImage(src_path_to_ref_img)

            # resample
            resample = sitk.ResampleImageFilter()
            resample.SetReferenceImage(ref_img)
            resample.SetInterpolator(sitk.sitkLinear) # other options: sitk.sitkLinear, sitk.sitkNearestNeighbor, sitk.sitkBSpline
            resampled_image = resample.Execute(raw_mask_img) # Run the resampling

            # set all voxels with value equal or greater then five to one and the rest to zero
            resampled_image[resampled_image < 5] = 0
            resampled_image[resampled_image >= 5] = 1   

            # save the resampled mask
            sitk.WriteImage(resampled_image, dst_path_to_resampled_mask)
        
        print(f"\tSuccessful resampling of mask {mask_name}.")



    def run_preprocessing(self):
        """
        Executes all the pre-processing steps, except the resampling, by calling the functions needed. 
        """
        self.sort_files_by_slice_number_and_bval()

        self.create_bvalsfile()
        self.create_dwi_4d_as_nifti()

        self.bval_series_as_nifti()





    
           
        




