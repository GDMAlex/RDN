from .base import BaseFE

class BinaryFE(BaseFE):
    def __init__(self):
        super().__init__()
        #### TODO: initialize the binary feature transformation
        self.binaire = 0
        
    
    def fit(self, data):
        #### TODO: find parameters for the binary feature transformation
        ##print("data = " , data)
        for i in data : 
            if i >= 0.5:
                self.binaire = 1 
            else :
                self.binaire = 0 
    
    def transform(self, data):
        #### TODO: apply the binary feature transformation to new data
        return self.binaire.astype(np.float32)