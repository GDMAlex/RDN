# commentaire de test
import libTP
import pandas

###### The objective of this step is to create a configuration JSON file that will be used
###### to setup the autoencoder structure, as well as find and save the parameters for 
###### the feature engineering functions
conf = {
    # This will store the name of the used attributes and their types 
    "attributes": {},
    # Store the parameters that will be used to build the network structure (ex: input size)
    "network": {}
}

###### TODO: Load the "train.csv" file using pandas
dataframe = pandas.read_csv("/Users/grandmaison/Desktop/TP/dataset/train.csv")

###### TODO: explore data and define the type (categorical, numerical or binary) of each attribute
######       Note: you can use the 'columns' member of pandas.DataFrame to list the attributes
######       and the dataset description to guide you (UNSW-NB15_features.csv)

###### Replace [...] with a list of all categorical attributes
for attr in dataframe.columns(1):
    conf["attributes"][attr] = "categorical"

###### Replace [...] with a list of all numerical attributes
for attr in dataframe.columns(2):
    conf["attributes"][attr] = "numerical"

###### Replace [...] with a list of all binary attributes
for attr in dataframe.columns(37):
    conf["attributes"][attr] = "binary"

###### TODO: create the transformation functions and find their parameters
######       Before that, you should complete the code inside libTP/feature_engineering folder
######       An example is provided for the categorical feature encoder
######       transforms is a dictionary of the form {attribute_name: feature_encoder_object}
transforms = {}


###### TODO: Set parameters required to build the network's structure
######       Expected format is as follow:
######       conf["network"] = {
######          attribute_name:  {
######            "type": attribute_type,
######            "output_size": ...,
######            "param1": ...
######            etc...
######       }


###### TODO: Don't forget to save:
######       - The conf inside conf/config.json
######         You can use python's built-in json module to do so
######       - The fitted feature encoder objects
######         The function is already implemented inside libTP.feature_engineering.helpers module  


