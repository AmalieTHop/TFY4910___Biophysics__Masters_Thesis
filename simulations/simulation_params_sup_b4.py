"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""



import numpy as np

class simulation_params_sup_b4:
    def __init__(self):
        self.bvalues = np.array([0, 50, 100, 800])
        self.num_samples_training = 1000000
        self.num_samples_test = 100000
        self.repeats = 25
        self.rician = True
        self.ranges = np.array([[0.0005, 0.05, 0.005], [0.003, 0.50, 0.1]])
        self.learning = 'sup'

