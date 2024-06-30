

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
