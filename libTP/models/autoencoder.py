from pytorch_lightning.core.lightning import LightningModule
from .encoder import Encoder
from .decoder import Decoder
from .loss import Loss
import torch
import os
import pickle


class AutoEncoder(LightningModule):
    def __init__(self, input_variables_dict, latent_dim=32):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(input_variables_dict, output_dim=self.hparams.latent_dim)
        
        self.decoder = Decoder(input_variables_dict, input_dim=self.hparams.latent_dim)

        self.loss = Loss(input_variables_dict)

        # Most likely, the scorer's parameters wil be a dictionary,
        # with one key for each attribute
        self.scorer_parameters = dict()

        ### TODO Suggested improvements:
        ##### Add Activations
        ##### Add Regularisation (ex: dropout, l1/l2 regularisation, ...)
        ##### Add Layers between encoder and decoder
        ##### ...
    
    def save_scorer(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        with open("%s/scorer"%path, 'wb') as f:
            pickle.dump(self.scorer_parameters, f)

    def load_scorer(self, path):
        with open("%s/scorer"%path, 'rb') as f:
            self.scorer_parameters = pickle.load(f)


    def forward(self, x):
        reprod = self.autoencoder(x)
        loss = self.loss(reprod, x)
        return self.score(loss)

    def autoencoder(self, x):
        encoded = self.encoder(x)
        # If you add layers, don't forget to call them here
        decoded = self.decoder(encoded)

        return decoded

    def score(self, loss_dict):
        # TODO: Scale each loss and combine them to output an anomaly score per sample 
        scaled_loss = {}
        for attr in loss_dict:
            # TODO: scale loss_dict[attr] before assigning it
            scaled_loss[attr] = (loss_dict[attr]-self.scorer_parameters[attr]["mu"])/self.scorer_parameters[attr]["sigma"]
            scaled_loss[attr] = torch.clip(scaled_loss[attr], 0, 10)
            
        return torch.sum(torch.stack([v for k, v in scaled_loss.items()]), dim=0)

    def calibration_step(self, batch):
        reprod = self.autoencoder(batch)
        return self.loss(reprod, batch)

    def calibrate(self, data_loader):
        self.eval()
        with torch.no_grad():
            # Compute loss for each batch
            batches_loss = [self.calibration_step(batch) for batch in data_loader]
            # Concatenate all the losses
            losses = {attr: torch.cat([batch_loss[attr] for batch_loss in batches_loss]) for attr in self.hparams.input_variables_dict}

            # TODO: Find the parameters for the scoring function using 'losses'
            for attr in losses:
                # TODO: calculate parameters here and store them inside self.scorer_parameters 
                self.scorer_parameters[attr] = {
                    'mu': losses[attr].mean(),
                    'sigma': losses[attr].std()
                }
    
    def training_step(self, batch, batch_idx):
        reprod = self.autoencoder(batch)
        loss_dict = self.loss(reprod, batch)
        
        # Sum all the losses
        loss = torch.stack([v for _, v in loss_dict.items()]).sum(dim=0)
        # Note: if you add l1/l2 regularisation you will have to add it in the loss

        # Log loss for each attribute
        for attr in loss_dict:         
            self.log("%s_loss"%attr, loss_dict[attr].mean())
        
        return loss.mean()

    
    def validation_step(self, batch, batch_idx):
        reprod = self.autoencoder(batch)
        loss_dict = self.loss(reprod, batch)
        # Sum all the losses
        val_loss = torch.stack([v for _, v in loss_dict.items()]).sum(dim=0).mean()

        # Log loss for each attribute
        for attr in loss_dict:         
            self.log("%s_val_loss"%attr, loss_dict[attr].mean())
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)