import libTP
import pandas
import pytorch_lightning

###### TODO: load configuration created at the previous step
conf = ...
###### TODO: load feature encoder created at the previous step
######       You can use libTP.feature_engineering.helpers
transforms = ...
###### TODO: load "train.csv" using pandas
data = ...

###### TODO: Complete the code in libTP/models 
model = libTP.models.AutoEncoder(conf["network"])

###### TODO: transform dataframe (you can use libTP.feature_engineering.helpers)

###### TODO: train the model. You can use functions inside libTP.misc.dataset to create
######       a torch.Dataset complient with what pytorch_lightning is expecting
######       Don't forget to split the dataset into 3 parts: train, evaluation, test
######       This can be done using the split method of class libTP.misc.dataset.PandasDataset


###### TODO: use the test data to calibrate the scoring functions of the autoencoder

###### TODO: save the trained model and go on to the next step