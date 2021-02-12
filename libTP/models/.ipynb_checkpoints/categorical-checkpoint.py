import torch


# TODO define the CategoricalEncoder
class CategoricalEncoder(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.layer = torch.nn.Sequential( torch.nn.Embedding(params["voc_size"], params["output_size"]), torch.nn.Sigmoid())
    
    def forward(self, x):
        return self.layer(x).squeeze()

# TODO define the CategoricalDecoder
class CategoricalDecoder(torch.nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.layer = torch.nn.Sequential( torch.nn.Embedding(params["output_size"], params["voc_size"]), torch.nn.Softmax())
    
    def forward(self, x):
        return self.layer(x).squeeze()

# TODO define the CategoricalLoss
class CategoricalLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, yh, y):
        yh = self(x) ## ou self.layer(x) 
        loss = self.loss(yh,y)
        return loss