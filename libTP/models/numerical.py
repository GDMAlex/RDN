import torch


# TODO define the NumericalEncoder
class NumericalEncoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer = torch.nn.Sequential( torch.nn.Linear(1, params["output_size"]), torch.nn.Sigmoid() )
        
    def forward(self, x):
        return self.layer(x)

# TODO define the NumericalDecoder
class NumericalDecoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.layer = torch.nn.Sequential( torch.nn.Linear(params["output_size"],1), torch.nn.ReLU() )
    def forward(self, x):
        return self.layer(x)

# TODO define the NumericalLoss
class NumericalLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.loss = torch.nn.MSELoss() 
    
    def forward(self, yh, y):
        yh = self(x)
        loss = self.loss(yh,y)
        return loss