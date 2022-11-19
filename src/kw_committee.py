# kw committee machine

# standard lib
from pathlib import Path

# common numerical and scientific libraries
import numpy as np

# pytorch
import torch
from torch import nn

import sys
sys.path.insert(0, str('../training'))
from cm_dataset import CommitteeDataset

class KWCommitteeModel(nn.Module):
    def __init__(self, n_features, dropout=0.1):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(n_features, 128)
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 2)
        self.dropout0 = nn.Dropout(p=dropout)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.dropout0(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.lin3(x)
        #x = self.softmax(x)
        return x

    @property
    def n_features(self):
        return self.lin0.in_features


class KWCommitteeMachine:
    """Committee machine with 5 cv-folds of the same model"""

    def __init__(self,
            weights_dir,    # if str or Path: load models from this directory
                            # if None: leave models uninitialized
            device='cpu',   # 'cpu' or 'cuda'
            ):
        self.device = torch.device(device)
        self.models = None
        if weights_dir is not None:
            self.build_and_load(weights_dir)

    @classmethod
    def model_path(cls, weights_dir, fold):
        return Path(weights_dir) / f'weights_cm_f{fold}.pt'

    def build_and_load(self, weights_dir):
        """build 5 models and load weights"""
        self.models = []
        for fold in range(5):
            path = self.model_path(weights_dir, fold)
            if not path.exists():
                print(f'skip {path}')
                self.models.append(None)
                continue
            print(f'load {path} ...')
            contents = torch.load(path)
            model_state_dict = contents.pop('model_state_dict')
            model = KWCommitteeModel(**contents)
            model.to(self.device)
            model.load_state_dict(model_state_dict, strict=True)
            self.models.append(model)


    def predict(self,
            data,   # shape=(n_samples, n_features)
            noise:float = 0.0,
            ):
        """return prediction as np.array, shape=(n_samples,). avg over 5 folds"""
        result = []
        for model in self.models:
            result.append(self.predict_model(model, data, noise))
        return np.stack(result, axis=-1).mean(axis=-1)

    def predict_model(self,
            model,
            data,   # shape=(n_samples, n_features)
            noise:float = 0.0
            ):
        """return prediction as np.array, shape=(n_samples,)"""
        data = torch.tensor(data).to(self.device)
        dset = CommitteeDataset(noise, data)
        loader = torch.utils.data.DataLoader(dset,
                batch_size = 3,
                )
        softmax = []
        model.eval()
        with torch.no_grad():
            for data, in loader:
                data = data.to(self.device)
                softmax_batch = nn.Softmax(dim=1)(model(data))
                softmax.append(softmax_batch.detach().cpu())
        softmax = torch.cat(softmax).numpy()
        return softmax[:,1]

# vim: set sw=4 sts=4 expandtab :
