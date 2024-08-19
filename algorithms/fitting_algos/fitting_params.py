"""
June 2024 by Amalie Toftum Hop
https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis

Code is uploaded as part of a Master’s thesis: 
Amalie Toftum Hop. “Deep Learning-Based Intravoxel Incoherent Motion Modelling of
Diffusion-Weighted MRI in Head and Neck Cancer: In Silico and In Vivo Studies.
Master thesis. Norwegian University of Science and Technology, 2024.
"""

class lsq_params:
    def __init__(self):
        self.method = 'lsq'
        self.do_fit = True
        self.fitS0 = True
        self.bounds = ([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])


class seg_params:
    def __init__(self):
        self.method = 'seg'
        self.do_fit = True  
        self.fitS0 = True
        self.bounds = ([0, 0, 0.005, 0.7], [0.005, 0.7, 0.3, 1.3])
        self.cutoff = 200