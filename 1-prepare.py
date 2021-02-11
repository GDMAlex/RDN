import libTP
import pandas as pd
import numpy as np
#test D'ALEX 

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
dataframe = pd.read_csv("Users/grandmaison/Desktop/RDN/dataset/train.csv")

###### TODO: explore data and define the type (categorical, numerical or binary) of each attribute
######       Note: you can use the 'columns' member of pandas.DataFrame to list the attributes
######       and the dataset description to guide you (UNSW-NB15_features.csv)

for attr in ["proto","service","state"]:
    conf["attributes"][attr] = "categorical"

###### Replace [...] with a list of all numerical attributes
for attr in ["dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl","dload","sloss","sinpkt","dinpkt","synack","ackdat","smean","dmean"
,"response_body_len","ct_srv_src","ct_dst_ltm","ct_src_dport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd","ct_src_ltm","ct_srv_dst"]:
    conf["attributes"][attr] = "numerical"

###### Replace [...] with a list of all binary attributes  
for attr in ["trans_depth","ct_flw_http_mthd","is_sm_ips_ports"]:
    conf["attributes"][attr] = "binary"

###### TODO: create the transformation functions and find their parameters
######       Before that, you should complete the code inside libTP/feature_engineering folder
######       An example is provided for the categorical feature encoder
######       transforms is a dictionary of the form {attribute_name: feature_encoder_object}
transforms = {}

for attr, t in conf["attributes"].items():

    if t == "categorical":
        transforms[attr] = libTP.feature_engineering.CategoricalFE()
    if t == "numerical":
        transforms[attr] = libTP.feature_engineering.NumericalFE()
    if t == "binary":
        transforms[attr] = libTP.feature_engineering.BinaryFE()

    transforms[attr].fit(dataframe[attr])


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


