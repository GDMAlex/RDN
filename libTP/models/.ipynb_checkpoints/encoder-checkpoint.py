import torch
from .categorical import CategoricalEncoder
from .numerical import NumericalEncoder
from .binary import BinaryEncoder

class Encoder(torch.nn.Module):
    def __init__(self, input_variables_dict, output_dim=32):
        super().__init__()
        self.attribute_encoders = torch.nn.ModuleDict()
        self.output_size = output_dim
        concatenated_size = 0
        for (attr, params) in input_variables_dict.items():
            ### The 'output_size' of each embedding function is important 
            ### to create the remaining structure of the encoder
            concatenated_size += params["output_size"]
            if params["type"] == "categorical":
                self.attribute_encoders[attr] = CategoricalEncoder(params)
            elif params["type"] == "binary":
                self.attribute_encoders[attr] = BinaryEncoder(params)
            elif params["type"] == "numerical":
                self.attribute_encoders[attr] = NumericalEncoder(params)
            else:
                raise NotImplementedError("Unsupported attribute type %s"%params["type"])
        
        # TODO: Improve this part of the encoder
        self.output = torch.nn.Linear(concatenated_size, output_dim)

    def forward(self, inputs):
        encoded = []

        for attr in self.attribute_encoders:
            encoded.append(self.attribute_encoders[attr](inputs[attr]))
        
        encoded_representation = torch.cat(encoded, dim=-1)
   
        return self.output(encoded_representation)


