#!/usr/bin/env python3

# standard library
import sys
import random
import datetime
from pathlib import Path
self_dir = Path(__file__).parent
#
import time


# common numerical and scientific libraries
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.optim as optim

# other common libraries
from tqdm import tqdm

# local
from kw_models import Effnet_Melanoma, Resnest_Melanoma, Seresnext_Melanoma
from kw_dataset import generate_df, get_transforms, MelanomaDataset
from kw_util import GradualWarmupSchedulerV2
from util import write_csv, write_pickle
sys.path.insert(0, str(self_dir / '../src'))
from kw_classifier import KWClassifier
from sam import SAM, SAM_AMP

# global parameters
default_n_workers = 8

def set_torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



class TrainableKWClassifier(KWClassifier):
    """5 cv-folds of the same convnet
    
    "single model": one fold
    "joint": (eg. in prediction): average for the 5 folds
    """

    @classmethod
    def log(cls, path, content):
        print(content)
        if path is not None:
            with open(path, 'a') as f:
                print(content, file=f)

    def train(self,
            out_dir,
            dsd: pd.DataFrame,
            data_root,
            batch_size:int,
            is_weighted:bool = False,
            n_epochs:int = None,        # if None: use model-spec default, eg. 15
            n_workers:int = default_n_workers,
            folds = (0,1,2,3,4),    # or eg. (0,) for only the 0-th fold
            use_amp:bool = True,
            use_aug:bool = True,
            fix_lr:bool = False,
            weight_decay:float = 0.,
            seed:int = 0,
            sam:bool = False,
            sam_rho:float = 0.05,
            sam_adaptive:bool = False,
            ):
        """train selected folds"""
        # process args
        out_dir = Path(out_dir)
        data_root = Path(data_root)
        if n_epochs is None:
            n_epochs = self.default_n_epochs
        #
        out_dir.mkdir(parents=True, exist_ok=True)
        set_torch_seed(seed)
        # load data
        df, meta_features = generate_df(dsd, self.use_meta,
                self.data_folder, self.out_dim, self.diagnosis2idx, data_root, is_training=True)
        print(f'meta_features={meta_features}')
        print(f'df={len(df)}')
        # loss fn
        if is_weighted:
            target = list(df['target'].values)
            # add 1-1 of missing categories, so class_w is proper shape
            for cat in range(self.out_dim):
                if cat not in target:
                    target.append(cat)
            class_w = compute_class_weight('balanced',
                    classes=np.unique(target), y=target)
            class_w = torch.tensor(class_w.astype(np.float32)).to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_w)
            print('class weights:', class_w)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        ##
        transforms_train, transforms_val = get_transforms(self.image_size)
        if not use_aug:
            transforms_train = transforms_val
        for fold in folds:
            self._train_fold(out_dir, fold, df, meta_features,
                    transforms_train, transforms_val, 
                    n_epochs = n_epochs,
                    batch_size = batch_size,
                    n_workers = n_workers,
                    use_amp = use_amp,
                    fix_lr = fix_lr,
                    weight_decay = weight_decay,
                    sam = sam,
                    sam_rho = sam_rho,
                    sam_adaptive = sam_adaptive,
                    )

    def _train_fold(self,
            out_dir:Path,
            fold:int,
            df,
            meta_features,
            transforms_train,
            transforms_val,
            n_epochs:int,
            batch_size:int, 
            n_workers:int,
            use_amp:bool,
            fix_lr:bool,
            weight_decay:float,
            sam:bool,
            sam_rho:float,
            sam_adaptive:bool,
            ):
        """train single fold. Do not call directly, use via train()"""
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f'train_m{self.model_n}_f{fold}.log'

        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]
        print(f'train_fold({fold}): train={len(df_train)}, valid={len(df_valid)}')

        dataset_train = MelanomaDataset(df_train, 'train', meta_features,
                transform=transforms_train)
        dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features,
                transform=transforms_val)
        train_loader = torch.utils.data.DataLoader(dataset_train,
                batch_size=batch_size,
                sampler=RandomSampler(dataset_train),
                num_workers=n_workers,
                )
        valid_loader = torch.utils.data.DataLoader(dataset_valid,
                batch_size=batch_size,
                num_workers=n_workers,
                )

        if self.models is None:
            model = self._build_model(pretrained=True)
        else:
            model = self.models[fold]

        model = model.to(self.device)

        auc_max = 0.
        auc_20_max = 0.
        model_file  = out_dir / f'weights_m{self.model_n}_f{fold}_best.pth'
        model_file2 = out_dir / f'weights_m{self.model_n}_f{fold}_best20.pth'
        model_file3 = out_dir / f'weights_m{self.model_n}_f{fold}_final.pth'

        sam_cls = SAM_AMP if use_amp else SAM

        base_optim = torch.optim.Adam
        #base_optim = torch.optim.AdamW
        if sam:
            #optimizer = SAM(model.parameters(), base_optim, lr=self.init_lr, rho=sam_rho, adaptive=sam_adaptive)
            #base_optim = torch.optim.SGD
            optimizer = sam_cls(model.parameters(), base_optim, lr=self.init_lr,
                    weight_decay=weight_decay, rho=sam_rho,
                    adaptive=sam_adaptive)
        else:
            optimizer = base_optim(model.parameters(), lr=self.init_lr,
                    weight_decay=weight_decay)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        if not fix_lr:
            scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, n_epochs - 1)
            scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10,
                    total_epoch=1, after_scheduler=scheduler_cosine)

        self.log(log_path,
                f'{"ep":3s} {"date_time":19s} {"lrate":8s} {"tr_loss":7s} {"va_loss":7s} {"va_acc":7s} {"va_auc":7s} {"va_auc20":7s}')
        for epoch in range(1, n_epochs + 1):

            label = f'f{fold} ep={epoch:<2d}'
            if sam:
                train_loss = self.train_epoch(model=model, loader=train_loader, optimizer=optimizer,
                    step=self.sam_step, scaler=scaler, use_amp=use_amp, label=label)
            else:
                train_loss = self.train_epoch(model=model, loader=train_loader, optimizer=optimizer,
                    step=self.step, scaler=scaler, use_amp=use_amp, label=label)
            try:
                is_ext = df_valid['is_ext'].values
            except KeyError:
                is_ext = None
            valid_loss, acc, auc, auc_20 = self.val_epoch(model, valid_loader,
                    is_ext=is_ext)

            self.log(log_path,
                    f'{epoch:3d} {datetime.datetime.now().isoformat(sep="_", timespec="seconds")} {optimizer.param_groups[0]["lr"]:.2e} {train_loss:7.5f} {valid_loss:7.5f} {acc/100:7.5f} {auc:7.5f} {auc_20:7.5f}')

            if not fix_lr:
                scheduler_warmup.step()    
                if epoch==2: scheduler_warmup.step() # bug workaround   
                
            if auc > auc_max:
                print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
                torch.save(model.state_dict(), model_file)
                auc_max = auc
            if auc_20 > auc_20_max:
                print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
                torch.save(model.state_dict(), model_file2)
                auc_20_max = auc_20

        torch.save(model.state_dict(), model_file3)

    def train_epoch(self, model, loader, optimizer, step, scaler=None, use_amp:bool=False,
            label:str='', scheduler=None):

        model.train()
        train_loss = []
        bar = tqdm(loader)
        for (data, target) in bar:
            loss = step(model=model, optimizer=optimizer, scaler=scaler, use_amp=use_amp,
                data=data, target=target)
            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description(
                    f'{label} loss={loss_np:.4f} smth={smooth_loss:.4f}')

        train_loss = np.mean(train_loss)
        return train_loss

    def step(self, model, optimizer, scaler, use_amp:bool, data, target):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            data, meta = data
            data, meta, target = (
                data.to(self.device),
                meta.to(self.device),
                target.to(self.device))
            logits = model(data, meta)
            loss = self.loss_fn(logits, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if self.image_size in [896, 576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        scaler.step(optimizer)
        scaler.update()
        return loss


    def sam_step(self, model, optimizer, scaler, use_amp:bool, data, target):
        if use_amp:
            return self._sam_amp_step(model, optimizer, scaler, data, target)
        data, meta = data
        data, meta, target = (
            data.to(self.device),
            meta.to(self.device),
            target.to(self.device))

        logits = model(data, meta)
        loss = self.loss_fn(logits, target)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        ## disable batch norm
        #for module in model.modules():
        #    if isinstance(module, nn.BatchNorm2d):
        #      module.eval()

        logits2 = model(data, meta)
        loss2 = self.loss_fn(logits2, target)
        loss2.backward()
        optimizer.second_step(zero_grad=True)

        ## enable batch norm 
        #model.train()

        return loss

    def _sam_amp_step(self, model, optimizer, scaler, data, target):
        use_amp = True

        with torch.cuda.amp.autocast(enabled=use_amp):
            data, meta = data
            data, meta, target = (
                data.to(self.device),
                meta.to(self.device),
                target.to(self.device))
            logits = model(data, meta)
            loss = self.loss_fn(logits, target)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer.base_optimizer)
        # were any infs?
        optimizer_state = scaler._per_optimizer_states[
                id(optimizer.base_optimizer)]
        were_infs = bool(sum(v.item()
                for v in optimizer_state["found_inf_per_device"].values()))
        if were_infs:
            # skip everything for this minibatch
            scaler.step(optimizer.base_optimizer)   # does not step optimizer
            scaler.update()
            return loss
        #
        optimizer.first_step(zero_grad=True)
        scaler.update()

        ## disable batch norm
        #for module in model.modules():
        #    if isinstance(module, nn.BatchNorm2d):
        #      module.eval()

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits2 = model(data, meta)
            loss2 = self.loss_fn(logits2, target)
        scaler.scale(loss2).backward()
        scaler.unscale_(optimizer.base_optimizer)
        optimizer.second_step(scaler, optimizer.base_optimizer, zero_grad=True)
        scaler.update()

        ## enable batch norm 
        #model.train()

        return loss

    def val_epoch(self, model, loader, is_ext=None, n_test=1):

        model.eval()
        val_loss = []
        LOGITS = []
        PROBS = []
        TARGETS = []
        bar = tqdm(loader)
        with torch.no_grad():
            for (data, target) in bar:
                data, meta = data
                data, meta, target = (data.to(self.device),
                        meta.to(self.device),
                        target.to(self.device))
                logits = torch.zeros(
                        (data.shape[0], self.out_dim)).to(self.device)
                probs = torch.zeros(
                        (data.shape[0], self.out_dim)).to(self.device)
                for I in range(n_test):
                    l = model(self.get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
                logits /= n_test
                probs /= n_test

                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())

                loss = self.loss_fn(logits, target)
                val_loss.append(loss.detach().cpu().numpy())
                bar.set_description('validation')

        val_loss = np.mean(val_loss)
        LOGITS = torch.cat(LOGITS).numpy()
        PROBS = torch.cat(PROBS).numpy()
        TARGETS = torch.cat(TARGETS).numpy()

        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        try:
            auc = roc_auc_score(
                    (np.isin(TARGETS, self.mel_idx)).astype(float), PROBS[:, self.mel_idx].sum(1))
        except ValueError:
            auc = np.nan
        try:
            auc_20 = roc_auc_score(
                    (TARGETS[is_ext == 0] == self.mel_idx).astype(float),
                    PROBS[is_ext == 0, self.mel_idx])
        except Exception:
            auc_20 = np.nan
        return val_loss, acc, auc, auc_20

    def predict_oof(self,
            dsd: pd.DataFrame,
            data_root,
            batch_size,
            n_workers,
            models = None,  # default: self.models
            n_tta = 8, # number of flip/transpose trials for test time augment
            out_dir = None,
            ):
        """predict out-of-fold"""
        # check arguments
        assert 'fold' in dsd.columns
        assert dsd['fold'].min() >= 0
        assert dsd['fold'].max() < len(self.models)
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        if models is None:
            models = self.models
        #
        if self.featurepred:
            out = pd.DataFrame({
                'name': dsd['name'],
                #'pred': [np.nan for n_ in dsd['name']]
                })
        else:
            out = pd.DataFrame({
                'name': dsd['name'],
                'pred': [np.nan for n_ in dsd['name']]
                })
                
        for fold in range(len(self.models)):
            fold_filter = dsd['fold'] == fold
            dsd_fold = dsd[fold_filter]
            model = models[fold]
            if model is None:
                continue
            ret = self.predict(dsd_fold, data_root, batch_size, n_workers,
                    models=[model], n_tta=n_tta,
                    out_dir=None)
            
            if self.featurepred:
                keys = list(ret.keys())
                feature_keys = [x for x in keys if x.startswith('feature')]
                feature_keys.sort()            
                for feature in feature_keys:
                    out.loc[fold_filter, (f'{feature}',)] = ret[f'{feature}'].values
            else:
                out.loc[fold_filter, ('pred',)] = ret['pred'].values                
                if any(colname.startswith('d7ft_') for colname in list(ret.columns)):
                    if not any(colname.startswith('d7ft_') for colname in list(out.columns)):
                        keys = [colname for colname in list(ret.columns) if colname.startswith('d7ft_')]
                        nans = [[np.nan for i_ in dsd['name']] for _ in range(len(keys))]
                        out.join(
                            pd.DataFrame({
                                **dict(zip(keys, nans))
                            })
                        )
                    for colname in list(ret.columns):
                        if colname.startswith('d7ft_'):
                            out.loc[fold_filter, colname] = ret[colname].values

        if 'target' in dsd.columns:
            out['true'] = dsd['target']
        out.dropna(inplace=True)
        out.set_index('name', inplace=True)
        
        # save cvs
        if out_dir:
            filename = 'feature_m' if self.featurepred else 'softmax_m'
            path = out_dir / f'{filename}{self.model_n}.csv'
            write_csv(path, out)
            path = out_dir / f'{filename}{self.model_n}.pkl'
            write_pickle(path, out)

        return out

    def predict(self,
            dsd: pd.DataFrame,
            data_root,
            batch_size,
            n_workers,
            models = None,  # default: self.models
            n_tta = 8, # number of flip/transpose trials for test time augment
            out_dir = None,
            ):
        """predict (default: using all 5 folds)"""
        # process args
        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        data_root = Path(data_root)

        df_test, meta_features = generate_df(dsd, self.use_meta,
                self.data_folder, self.diagnosis2idx, self.out_dim, data_root)
        transforms_train, transforms_val = get_transforms(self.image_size)

        dataset_test = MelanomaDataset(df_test, 'test', meta_features,
                transform=transforms_val)
        test_loader = torch.utils.data.DataLoader(dataset_test,
                batch_size=batch_size,
                num_workers=n_workers,
                )

        # predict
        softmax_all = self._predict(test_loader, models=models, n_tta=n_tta,
                use_tqdm=True)

        # output
        if self.featurepred:
            out = pd.DataFrame({
                'name': dsd['name'],                
                })
            for i in range(softmax_all.shape[1]):
                out[f'feature_{i}'] = softmax_all[:,i]
        else:
            if softmax_all.shape[1] > 2:
                feature_names = [f'd7ft_{i}' for i in range(softmax_all.shape[1])]
                key_val = dict(zip(feature_names, softmax_all.T))
                out = pd.DataFrame({
                'name': df_test['image_name'],
                'pred': softmax_all[:, self.mel_idx].sum(1),
                **key_val
                })
            else:
                out = pd.DataFrame({
                'name': df_test['image_name'],
                'pred': softmax_all[:, self.mel_idx].sum(1),
                })
        # save features    

        if 'target' in dsd.columns:
            out['true'] = dsd['target']
        out.set_index('name', inplace=True)

        # save results
        if out_dir:
            filename = 'feature_m' if self.featurepred else 'softmax_m'
            path = out_dir / f'{filename}{self.model_n}.csv'
            write_csv(path, out)
            path = out_dir / f'{filename}{self.model_n}.pkl'
            write_pickle(path, out)

        return out
# vim: set sw=4 sts=4 expandtab :
