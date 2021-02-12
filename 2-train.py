import libTP
import pandas as pd 
import pytorch_lightning as pl
import json

###### TODO: load configuration created at the previous step
with open('./conf/config.json') as json_file:
    conf = json.load(json_file)
    
###### TODO: load feature encoder created at the previous step
######       You can use libTP.feature_engineering.helpers
transforms = libTP.feature_engineering.helpers.load_transforms(conf)
###### TODO: load "train.csv" using pandas
data = load.read_csv("./dataset/train.csv")

###### TODO: Complete the code in libTP/models 
model = libTP.models.AutoEncoder(conf["network"])

###### TODO: transform dataframe (you can use libTP.feature_engineering.helpers)

transformed = libTP.feature_engineering.helpers.transform_df(data,transforms)

###### TODO: train the model. You can use functions inside libTP.misc.dataset to create
######       a torch.Dataset complient with what pytorch_lightning is expecting
######       Don't forget to split the dataset into 3 parts: train, evaluation, test
######       This can be done using the split method of class libTP.misc.dataset.PandasDataset

early_stopping = pl.callbacks.EarlyStopping(min_delta=0.01, patience=10, monitor='val_loss')
checkpoints = pl.callbacks.ModelCheckpoint(monitor="val_loss", dirpath="./conf/checkpoints/")
trainer = pl.Trainer(callbacks=[early_stopping, checkpoints], max_epochs=1000)
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=validation_loader)
model = MyNet.load_from_checkpoint(checkpoints.best_model_path)
###### TODO: use the test data to calibrate the scoring functions of the autoencoder

###### TODO: save the trained model and go on to the next step