# learnable kw committee machine

# standard lib
import sys
import random
import datetime
import csv
from pathlib import Path
self_dir = Path(__file__).parent

# common numerical and scientific libraries
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# pytorch
import torch
from torch import nn

# other common libraries
from tqdm import tqdm

# local
from util import read_csv, write_csv
sys.path.insert(0, str(self_dir / '../src'))
from kw_committee import KWCommitteeModel, KWCommitteeMachine
from cm_dataset import CommitteeDataset
# 

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_split(path, label):
    """return: bool ndarray, shape=(n_samples,)"""
    df = read_csv(path)
    return np.array(df['split'] == label)

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_input(path:Path):
    """return: float32 ndarray, shape=(n_samples, dim)"""
    if path.suffix == '.csv':
        df = pd.read_csv(path)
        try:
            return df['pred'].values[:,None].astype(np.float32)
        except:
            return df['target'].values[:,None].astype(np.float32)
    elif path.suffix == '.h5':
        with h5py.File(path, 'r') as h5f:
            arr = h5f['pred'][()]
            if arr.shape[1] == 2:
                return arr[:,1][:,None].astype(np.float32)
            else:
                return arr.astype(np.float32)
    elif path.suffix == '.pkl':
        df = pd.read_pickle(path)
        return df.filter(like='feature').values.astype(np.float32)
    raise ValueError(path)

def load_inputs(path_list, is_v04=False):
    if not is_v04:
        collect = []
        for path in path_list:
            print('non-v04', path)
            collect.append(load_input(Path(path)))
        return np.concatenate(collect, axis=-1)
    # now v04
    names = None
    trues = None
    pred_collect = []
    for path in path_list:
        print(path)
        df = read_csv(path)
        if names is None:
            names = df['name'].values
            trues = df['true'].values
        else:
            #assert (names == df['name'].values).all()
            #assert (trues == df['true'].values).all()
            pass
        d7ftcolnanes = [colname for colname in list(df.columns) if colname.startswith('d7ft_')]
        if len(d7ftcolnanes) > 0:
            for col in d7ftcolnanes:
                pred_collect.append(df[col].values)
        else:
            pred_collect.append(df['pred'].values)
        print(d7ftcolnanes)
    return np.stack(pred_collect, axis=-1).astype(np.float32)


class TrainableKWCommitteeMachine(KWCommitteeMachine):

    @classmethod
    def log(cls, path, content):
        print(content)
        if path is not None:
            with open(path, 'a') as f:
                print(content, file=f)

    def train_predict(self,
            out_dir: Path,
            dsd: pd.DataFrame,
            data,   # shape=(n_samples, n_features)
            lr,
            n_epochs=100,
            batch_size=64,
            dropout=0.1,
            noise=0.0,
            init_to_avg = False,
            seed=0,
            ):
        if seed:
            set_torch_seed(seed)
        assert len(dsd) == len(data)
        n_features = data.shape[1]
        # n_folds
        n_folds = len(set(dsd['fold']))
        assert n_folds == dsd['fold'].max() + 1
        assert n_folds == 5 # for now
        #
        predictions = []
        for fold in range(n_folds):
            filt_train = (dsd['fold'] != fold)
            filt_valid = (dsd['fold'] == fold)
            # train
            x_train = data[filt_train]
            yt_train = np.array(dsd.loc[filt_train, 'target'])
            x_valid = data[filt_valid]
            yt_valid = np.array(dsd.loc[filt_valid, 'target'])
            assert x_train.shape[0] + x_valid.shape[0] == len(dsd)
            model = KWCommitteeModel(data.shape[1], dropout=dropout)
            if init_to_avg:
                eps = 1e-3
                sd = model.state_dict()
                sd['lin0.weight'][()] = eps * sd['lin0.weight'][()] + 1/sd['lin0.weight'].shape[1]
                sd['lin1.weight'][()] = eps * sd['lin1.weight'][()] + 1/sd['lin1.weight'].shape[1]
                sd['lin2.weight'][()] = eps * sd['lin2.weight'][()] + 1/sd['lin2.weight'].shape[1]
                sd['lin3.weight'][0,:] = eps * sd['lin3.weight'][0,:] - 1/sd['lin3.weight'].shape[1]
                sd['lin3.weight'][1,:] = eps * sd['lin3.weight'][1,:] + 1/sd['lin3.weight'].shape[1]

                sd['lin0.bias'][()] = 0
                sd['lin1.bias'][()] = 0
                sd['lin2.bias'][()] = 0
                sd['lin3.bias'][0] = 1
                sd['lin3.bias'][1] = 0
            self._train_fold(
                    model,
                    x_train,
                    yt_train,
                    valid_data = x_valid,
                    valid_target = yt_valid,
                    lr = lr,
                    n_epochs = n_epochs,
                    batch_size = batch_size,
                    noise = noise,
                    logfile = out_dir / 'train.log',
                    )
            # save
            path = self.model_path(out_dir, fold)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'n_features': n_features,
                'dropout': dropout,
                'model_state_dict': model.state_dict(),
                }, path)
            # predict
            predictions.append((filt_valid, self.predict_model(model, x_valid)))
        # save predictions
        df = pd.DataFrame({
            'name': dsd['name'],
            'true': dsd['target'],
            'pred': np.nan,
            })
        for filt, pred in predictions:
            df.loc[filt, 'pred'] = pred
        print('nan:', np.isnan(df['pred']).sum())
        path = out_dir / 'softmax_cm.csv'
        write_csv(path, df, index=False)

    def _train_fold(self,
            model,
            data,   # shape=(n_samples, n_features)
            target, # shape=(n_samples,)
            valid_data=None,
            valid_target=None,
            lr=1e-4,
            n_epochs=250,
            batch_size=64,
            noise=0.0,
            is_weighted=True,
            logfile=None,
            ):
        set_torch_seed()
        class_w = compute_class_weight('balanced', classes=np.unique(target),
                y=target)
        class_w = torch.tensor(class_w.astype(np.float32)).to(self.device)
        data = torch.tensor(data)
        target = torch.tensor(target)
        dset = CommitteeDataset(noise, data, target)
        train_loader = torch.utils.data.DataLoader(dset,
                batch_size=batch_size,
                )
        if valid_data is not None:
            valid_data = torch.tensor(valid_data)
            valid_target = torch.tensor(valid_target)
            dset_valid = CommitteeDataset(0,valid_data,valid_target)
            valid_loader = torch.utils.data.DataLoader(dset_valid,
                    batch_size=batch_size,
                    )
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if is_weighted:
            loss_fn = nn.CrossEntropyLoss(weight=class_w)
        else:
            loss_fn = nn.CrossEntropyLoss()

        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        self.log(logfile,
                f'{"ep":3s} {"date_time":19s} {"lrate":8s} ' + 
                f'{"tr_loss":7s} {"va_loss":7s} {"va_ba":7s}')
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer,
                    loss_fn)
            if valid_data is not None:
                va_loss, va_ba = self.valid_epoch(model, valid_loader,
                        loss_fn)
            else:
                va_loss = np.nan
                va_ba = np.nan
            #content = f'{epoch:3d} {datetime.datetime.now().isoformat(sep="_", timespec="seconds")} {optimizer.param_groups[0]["lr"]:.2e} {train_loss:.5f} {valid_loss:.5f} {acc/100:.5f} {auc:.5f} {auc_20:.5f}'
            self.log(logfile,
                    f'{epoch:3d} {datetime.datetime.now().isoformat(sep="_", timespec="seconds")} {optimizer.param_groups[0]["lr"]:.2e} ' + 
                    f'{train_loss:.5f} {va_loss:7.5f} {va_ba:7.5f}')


    def train_epoch(self, model, loader, optimizer, loss_fn):
        model.train()
        train_loss = []
        #bar = tqdm(loader)
        bar = loader
        for (data, target) in bar:
            optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
            pred = model(data)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        return np.mean(train_loss)

    def valid_epoch(self, model, loader, loss_fn):
        model.eval()
        loss_all = []
        target_all = []
        softmax_all = []
        bar = loader
        with torch.no_grad():
            for data, target in bar:
                target_all.append(target)
                data, target = data.to(self.device), target.to(self.device)
                pred = model(data)
                loss = loss_fn(pred, target)
                loss_all.append(loss.detach().cpu().numpy())
                softmax_all.append(
                        nn.Softmax(dim=1)(pred).detach().cpu())
        target_all = np.concatenate(target_all)
        softmax_all = torch.cat(softmax_all).numpy()
        ba = balanced_accuracy_score(target_all, softmax_all[:,1] > 0.5)
        return np.mean(loss_all), ba


# vim: set sw=4 sts=4 expandtab :
