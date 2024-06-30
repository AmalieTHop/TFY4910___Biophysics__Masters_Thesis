# TFY4910___Biophysics__Masters_Thesis


This repository is part of a master's thesis that was written as part of a Master's degree in Applied Physics and Mathematics with specialisation in Biophysics and Medical Technology at the Norwegian University of Science and Technology (NTNU). The work was carried out at the Department of Physics at NTNU during the spring semester of 2024 and is a part of the ongoing EMINENCE study at the St. Olavs Hospital, which aims to personalise radiotherapy for patients with head and neck cancer through improved functional imaging.


The complete source code is accessible from the GitHub repository https://github.com/AmalieTHop/TFY4910___Biophysics__Masters_Thesis. The code is written in Python version 3.10.13. The implementation of the algorithms LSQ, SEG, and DNN<sub>SSL</sub>, as well as the code that executes the simulations, are based on scripts available in public GitHub repositories. However, these scripts have been modified, adapted, and further developed to align with the methodology and objectives of this thesis. The original repositories will now be referenced. 

The LSQ and SEG algorithms were based on the scripts _LSQ\_fitting.py_ and _two\_step\_IVIM\_fit.py_ written by Oliver Gurney-Champion and Paulien Voorter, respectively. These scripts can be found in the GitHub repository https://github.com/OSIPI/TF2.4\_IVIM-MRI\_CodeCollection, maintained by the Open Science Initiative for Perfusion Imaging (OSIPI) [1].

The DNN<sub>SL</sub> algorithm and the workflow of the simulations were based on code built by Sebastiano Barbieri, which also has been a part of a publication by Kaandorp et al. [2]. The code can be found in the scripts _deep.py_, _hyperparams.py_, and _simulations.py_ from the GitHub repository https://github.com/oliverchampion/IVIMNET/tree/master [3]. The DNN<sub>SL</sub> algorithm was developed in-house by the author of this thesis and has been integrated into the functions used by DNN<sub>SSL</sub>. The code for DNN<sub>SSL</sub> served as a template for DNN<sub>SL</sub>. 


#### Training on slurm
Part of the code was run on the high-performance computing cluster IDUN provided by NTNU. The Python scripts managing the workflow of computations related to the cluster and their corresponding bash shell scripts have been added to the folder _cluster_.

#### Dependencies
Dependencies can be found in _Pipfile_.

#### Future
This repository is currently being modified. A more well-documented and user-friendly code will soon be released. 


#### References
[1] TF2.4_IVIM_code_collection. original-date: 2023-03-31T11:52:39Z. 2023. URL: https://github.com/OSIPI/TF2.4_IVIM-MRI_CodeCollection

[2] Misha P. T. Kaandorp et al. “Improved unsupervised physics-informed deep learning for intravoxel incoherent motion modeling and evaluation in pancreatic cancer patients”. In: Magnetic Resonance in Medicine 86.4 (2021), pp. 2250–2265. ISSN: 1522-2594. DOI: 10.1002/mrm.28852. URL: https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.28852

[3] Oliver Gurney-Champion. IVIMNET. original-date: 2020-10-22T08:33:51Z. 2023. URL: https://github.com/oliverchampion/IVIMNET


