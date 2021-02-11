import torch


# TODO define the BinaryEncoder
class BinaryEncoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer = Sequential( Linear(1, params["output_size"]),Sigmoid() )
        
    def forward(self, x):
        return self.layer(x)

# TODO define the BinaryDecoder
class BinaryDecoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.layer = Sequential( Linear(params["output_size"],1), Sigmoid() )
    
    def forward(self, x):
        return self.layer(x) 

# TODO define the BinaryLoss
class BinaryLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.loss = torch.nn.BCELoss() 
    
    def forward(self, yh, y):
        yh = self.layer(y)
        return yh