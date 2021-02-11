import torch


# TODO define the CategoricalEncoder
class CategoricalEncoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer = Sequential( Embedding(params["voc_size"], params["output_size"]), Sigmoid())
    
    def forward(self, x):
    return self.layer(x).squeeze()

# TODO define the CategoricalDecoder
class CategoricalDecoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.layer = Sequential( Embedding(params["output_size"], params["voc_size"]), softmax())
    
    def forward(self, x):
        return self.layer(x).squeeze()

# TODO define the CategoricalLoss
class CategoricalLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        pass
    
    def forward(self, yh, y):
        return 