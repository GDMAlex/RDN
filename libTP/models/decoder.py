import torch
from .categorical import CategoricalDecoder
from .numerical import NumericalDecoder
from .binary import BinaryDecoder

class Decoder(torch.nn.Module):
    def __init__(self, input_variables_dict, input_dim=32):
        super().__init__()
        self.attribute_decoders = torch.nn.ModuleDict()

        for (attr, params) in input_variables_dict.items():
            if params["type"] == "categorical":
                self.attribute_decoders[attr] = CategoricalDecoder(input_dim=input_dim, params=params)
            elif params["type"] == "binary":
                self.attribute_decoders[attr] = BinaryDecoder(input_dim=input_dim, params=params)
            elif params["type"] == "numerical":
                self.attribute_decoders[attr] = NumericalDecoder(input_dim=input_dim, params=params)
            else:
                raise NotImplementedError("Unsupported attribute type %s"%params["type"])
        
        

    def forward(self, x):
        
        decoded = dict()

        for attr in self.attribute_decoders:
            decoded[attr] = self.attribute_decoders[attr](x)              
        return decoded


