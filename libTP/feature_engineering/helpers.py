from .categorical import CategoricalFE
from .numerical import NumericalFE
from .binary import BinaryFE
import pandas as pd

### takes as input a dictionary in the form {attr: FeatureExtractor} and save to disk
def save_transforms(transforms, base_path="./conf/fe"):
    if not os.path.exists(path):
        os.makedirs(path)
    # TODO implement transforms saving/loading functions


### returns a dictionary in the form {attr: FeatureExtractor}
def load_transforms(conf, base_path='./conf/fe'):
    # TODO implement transforms saving/loading functions
    transforms = {}
    return transforms

### takes a dataframe and a dictionary in the form {attr: FeatureExtractor} as input 
### returns the transformed dataframe
def transform_df(df, transforms):
    res = pd.DataFrame()
    for attr, fe in transforms.items():
        res[attr] = fe(df[attr])
    return res