import pickle
import pandas as pd
import os

class BaseFE():
    def __init__(self, **kwargs):
        pass
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for k, v in data.items():
                setattr(self, k, v)
    
    def fit(self, data):
        raise NotImplementedError()
    
    def transform(self, data):
        raise NotImplementedError()

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def __call__(self, data):
        return self.transform(data)

