import numpy as np

class simulation_params:
    def __init__(self):
        self.bvalues = np.array([0, 10, 20, 30, 40, 50, 60, 100, 200, 500, 800])
        self.num_samples_training = 1000000 #10000000
        self.num_samples_test = 100000
        self.repeats = 3
        self.rician = True
        self.ranges = np.array([[0.0005, 0.05, 0.005], [0.003, 0.50, 0.1]])
        self.learning = 'sup'
        self.num_trials = 250
        
