import torch
import pandas

def dict_collate(batch):
    res = {}
    for row in batch:
        for k in row:
            res[k] = torch.cat( [res.get(k, torch.tensor([], dtype=row[k].dtype)), row[k]]) 
    
    return res

def batch_loader(dataset, batch_size=64, num_workers=4):
    sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size=batch_size, drop_last=False)
    loader = torch.utils.data.DataLoader(dataset, sampler=sampler, num_workers=num_workers, collate_fn=dict_collate)
    return loader 

class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, df, **kwargs):
        self.data = df
        
    def split(self, train=0.6, val=0.2, test=0.2, shuffle=True):
        df = self.data.sample(frac=1).reset_index(drop=True)
        
        train_stop = int(len(self.data)*train)
        val_stop = train_stop + int(len(self.data)*val)
        test_stop = val_stop + int(len(self.data)*test)

        return (PandasDataset(df[:train_stop]),
                PandasDataset(df[train_stop:val_stop]),
                PandasDataset(df[val_stop:test_stop]))    

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # If there is only one index, transform it into a list
        if isinstance(idx, int):
            idx = [idx]

        subset = self.data.iloc[idx]        
        return {k: torch.tensor(subset[k].values).unsqueeze(-1) for k in subset.columns}