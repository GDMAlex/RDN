from .base import BaseFE

class NumericalFE(BaseFE):
    def __init__(self):
        super().__init__()
        #### TODO: initialize the numerical feature transformation
        pass
    
    def fit(self, data):
        #### TODO: find parameters for the numerical feature transformation
        pass
    
    def transform(self, data):
        #### TODO: apply the numerical feature transformation to new data
        pass