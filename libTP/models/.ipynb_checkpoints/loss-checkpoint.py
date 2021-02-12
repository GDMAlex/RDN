import torch
from .categorical import CategoricalLoss
from .numerical import NumericalLoss
from .binary import BinaryLoss

class Loss(torch.nn.Module):
    def __init__(self, input_variables_dict):
        super().__init__()
        
        self.losses = torch.nn.ModuleDict()
        for (attr, params) in input_variables_dict.items():
            if params["type"] == "categorical":
                self.losses[attr] = CategoricalLoss(params)
            elif params["type"] == "binary":
                self.losses[attr] = BinaryLoss(params)
            elif params["type"] == "numerical":
                self.losses[attr] = NumericalLoss(params)
            else:
                raise NotImplementedError("Unsupported attribute type %s"%params["type"])
        

    def forward(self, yh, y):
        output = dict()
        for attr in self.losses:
            output[attr] = self.losses[attr](yh[attr], y[attr])
        
        return output


