from .categorical import CategoricalFE
from .numerical import NumericalFE
from .binary import BinaryFE
import pandas as pd
import os 
import numpy as np 

### takes as input a dictionary in the form {attr: FeatureExtractor} and save to disk
def save_transforms(transforms, base_path="./conf/fe"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    # TODO implement transforms saving/loading functions
    with open(base_path+'/test', 'wb') as f:
        np.save(f, transforms)

### returns a dictionary in the form {attr: FeatureExtractor}
def load_transforms(conf, base_path='./conf/fe'):
    # TODO implement transforms saving/loading functions
    transforms = {}
    with open(base_path+'/test', 'rb') as f:
        transforms = np.load(f,allow_pickle = True)
        
    #transforms = dict(enumerate(transforms.flatten(), 1))
    transforms = transforms.tolist() 
    #transforms.ravel() 
    #transforms.reshape(-1)
    #print(type(transforms))
    #print(transforms)
    return transforms

### takes a dataframe and a dictionary in the form {attr: FeatureExtractor} as input 
### returns the transformed dataframe
def transform_df(df, transforms):
    res = pd.DataFrame()
    for attr, fe in transforms.items():
        res[attr] = fe(df[attr])
    return res