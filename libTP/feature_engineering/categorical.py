from .base import BaseFE

class CategoricalFE(BaseFE):
    def __init__(self, max_size=None):
        super().__init__()
        self.association_dict = {}
        self.max_size = max_size

    def fit(self, data):
        if not isinstance(self.max_size, int):
            self.max_size = len(data.unique()) + 1
        # value_counts will sort unique values from the most frequent to the least
        for v in data.value_counts().index:
            # We don't want to store more associations than max_size allows
            if len(self.association_dict)+1 >= self.max_size:
                break
            
            # Associate an integer to each possible category
            if not v in self.association_dict:
                self.association_dict[v] = len(self.association_dict)
            
    def transform_fn(self, x):
        return self.association_dict[x] if x in self.association_dict else self.max_size-1
    
    def transform(self, data):
        return data.map(self.transform_fn)