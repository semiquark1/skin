import torch
from torch.utils.data import TensorDataset

class CommitteeDataset(TensorDataset):
    def __init__(self, eps:float = 0.0, *tensors):
        super().__init__(*tensors)
        self.eps = eps
        self.one_minus_eps = 1 - eps
        self.data_trf = self._identity if eps == 0.0 else self._noisy
    
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item = (self.data_trf(item[0]),*item[1:])
        return item
    
    def _identity(self, data):
        return data

    def _noisy(self, data):
        min_boundry = self.one_minus_eps * data
        max_boundry = min_boundry + self.eps
        interval = max_boundry - min_boundry
        noise = torch.rand(data.shape)
        return interval * noise + min_boundry