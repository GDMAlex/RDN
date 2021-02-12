from .base import BaseFE
import numpy as np

class NumericalFE(BaseFE):
    def __init__(self):
        super().__init__()
        self.mu = 0
        self.sigma = 0
        
    def fit(self, data):
        self.mu = data.mean()
        self.sigma = data.std()
        
    def transform(self, data):
        return ((data - self.mu)/self.sigma).astype(np.float32)