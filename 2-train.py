import libTP
import pandas as pd 
import pytorch_lightning as pl
import json

print(pl.__version__)

###### TODO: load configuration created at the previous step
with open('./conf/config.json') as json_file:
    conf = json.load(json_file)
    
###### TODO: load feature encoder created at the previous step
######       You can use libTP.feature_engineering.helpers
transforms = libTP.feature_engineering.helpers.load_transforms(conf)
###### TODO: load "train.csv" using pandas
data = pd.read_csv("./dataset/train.csv")

###### TODO: Complete the code in libTP/models 
model = libTP.models.AutoEncoder(conf["network"])

###### TODO: transform dataframe (you can use libTP.feature_engineering.helpers)
transformed = libTP.feature_engineering.helpers.transform_df(data,transforms)

###### TODO: train the model. You can use functions inside libTP.misc.dataset to create
######       a torch.Dataset complient with what pytorch_lightning is expecting
######       Don't forget to split the dataset into 3 parts: train, evaluation, test
######       This can be done using the split method of class libTP.misc.dataset.PandasDataset


#Split train , val , test
train , evaluation , test = libTP.misc.dataset.PandasDataset(transformed).split() #Split pour avoir train , evaluation et test 
#Batch sur train , val , test 
train_batch = libTP.misc.dataset.batch_loader(train)
evaluation_batch = libTP.misc.dataset.batch_loader(evaluation)
test_batch = libTP.misc.dataset.batch_loader(test)


#Entraînement
trainer = pl.Trainer(max_epochs=1000)
trainer.test(test_dataloaders=test_batch )
trainer.fit(model, train_dataloader=train_batch, val_dataloaders=evaluation_batch )


###### TODO: use the test data to calibrate the scoring functions of the autoencoder
###### TODO: save the trained model and go on to the next step