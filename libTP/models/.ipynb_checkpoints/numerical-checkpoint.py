import torch


# TODO define the NumericalEncoder
class NumericalEncoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer = Sequential( Linear(1, params["output_size"]),Sigmoid() )
        
    def forward(self, x):
        return self.layer(x)

# TODO define the NumericalDecoder
class NumericalDecoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.layer = Sequential( Linear(params["output_size"],1), ReLU() )
    def forward(self, x):
        return self.layer(x)

# TODO define the NumericalLoss
class NumericalLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.loss = torch.nn.MSELoss() 
    
    def forward(self, yh, y):
        yh = self.layer(y)
        return yh