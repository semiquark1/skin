# standard library
import io
from types import SimpleNamespace
from typing import List, Dict
from pathlib import Path

# common numerical and scientific libraries
import numpy as np
import pandas as pd

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations
import geffnet
import timm
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d

# other common libraries
from tqdm import tqdm

# derm7pt feature training support
import derm7pt_support

def get_modelspec(model_n):
    """return SimpleNamespace instance with attrs set"""
    data = pd.read_csv(io.StringIO("""
m  kernel_type                                   arxiv_m data imgsz init_lr
01 9c_meta_b3_768_512_ext_18ep                         1  768   512    3e-5
02 9c_b4ns_2e_896_ext_15ep                             5 1024   896    2e-5
03 9c_b4ns_448_ext_15ep-newfold                        6  512   448    3e-5
04 9c_b4ns_768_640_ext_15ep                            2  768   640    3e-5
05 9c_b4ns_768_768_ext_15ep                            3  768   768    3e-5
06 9c_meta_b4ns_640_ext_15ep                           4  768   640    3e-5
07 4c_b5ns_1.5e_640_ext_15ep                           9  768   640  1.5e-5
08 9c_b5ns_1.5e_640_ext_15ep                           8  768   640  1.5e-5
09 9c_b5ns_448_ext_15ep-newfold                       10  512   448    3e-5
10 9c_meta128_32_b5ns_384_ext_15ep                     7  512   384    3e-5
11 9c_b6ns_448_ext_15ep-newfold                       13  512   448    3e-5
12 9c_b6ns_576_ext_15ep_oldfold                       12  768   576    3e-5
13 9c_b6ns_640_ext_15ep                               11  768   640    3e-5
14 9c_b7ns_1e_576_ext_15ep_oldfold                    15  768   576    1e-5
15 9c_b7ns_1e_640_ext_15ep                            16  768   640    1e-5
16 9c_meta_1.5e-5_b7ns_384_ext_15ep                   14  512   384    3e-5
17 9c_nest101_2e_640_ext_15ep                         18  768   640    2e-5
18 9c_se_x101_640_ext_15ep                            17  768   640    3e-5
21 9c_b4ns_380_ext_15ep                                0  512   380    3e-5
22 9c_b4ns_456_15ep                                    0  512   456    3e-5
23 9c_b4ns_528_15ep                                    0  768   528    3e-5
32 2c_b4ns_380_ext_15ep_feature_blue_whitish_veil      0  512   380    3e-5
33 3c_b4ns_380_ext_15ep_feature_pigment_network        0  512   380    3e-5
34 3c_b4ns_380_ext_15ep_feature_streaks                0  512   380    3e-5
35 5c_b4ns_380_ext_15ep_feature_pigmentation           0  512   380    3e-5
36 4c_b4ns_380_ext_15ep_feature_regression_structures  0  512   380    3e-5
37 3c_b4ns_380_ext_15ep_feature_dots_and_globules      0  512   380    3e-5
38 8c_b4ns_380_ext_15ep_feature_vascular_structures    0  512   380    3e-5
40 9c_v2m_512_ext_15ep_test                            0  512   512    2e-5
41 9c_v2m_480_ext_15ep                                 0  512   480    2e-5
"""), delim_whitespace=True, dtype={'m':str}).set_index('m')
    return SimpleNamespace(**data.loc[model_n])


net_type_dict = {
        '_b3_':      'efficientnet_b3',
        '_b4ns_':    'tf_efficientnet_b4_ns',
        '_b5ns_':    'tf_efficientnet_b5_ns',
        '_b6ns_':    'tf_efficientnet_b6_ns',
        '_b7ns_':    'tf_efficientnet_b7_ns',
        '_nest101_': 'resnest101',
        '_se_x101_': 'seresnext101',
        '_v2s_':     'tf_efficientnetv2_s_in21k',
        '_v2m_':     'tf_efficientnetv2_m_in21k',
        '_v2l_':     'tf_efficientnetv2_l_in21k',
        }

######## dataset

class PredictDataset(Dataset):
    def __init__(self, images, metas, use_meta, image_size, data_folder):

        self.images = images
        self.metas = metas
        assert len(self.images) == len(self.metas)
        self.use_meta = use_meta
        self.data_folder = data_folder
        self.transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]

        res = self.transform(image=image)
        image = res['image'].astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            inp_meta = self.metas[index]
            sex = {'male': 1, 'female': 0}.get(inp_meta['sex'], -1)
            if inp_meta['age'] is not None:
                age_approx = inp_meta['age'] / 90
            else:
                age_approx = 0
            n_images = inp_meta.get('n_images', 1)
            n_images = np.log1p(n_images)
            image_size = {
                    512: 10.989,
                    768: 11.590,
                    1024: 0,    # no meta for data_folder==1024 model
                    }[self.data_folder]
            # site, if present
            site_value = inp_meta.get('site')
            try:
                site_i = ['anterior torso', 'head/neck', 'lateral torso',
                    'lower extremity', 'oral/genital', 'palms/soles',
                    'posterior torso', 'torso', 'upper extremity',
                    ].index(site_value)
            except ValueError:
                site_i = 9
            site = [0]*10
            site[site_i] = 1
            meta = [sex, age_approx, n_images, image_size] + site
        else:
            meta = 0.
        data = (torch.tensor(image).float(), torch.tensor(meta).float())

        return data


######## models

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, featurepred=False):
        super(Effnet_Melanoma, self).__init__()
        self.featurepred = featurepred
        self.n_meta_features = n_meta_features
        if 'efficientnetv2' in enet_type:
            self.enet = timm.create_model(enet_type, pretrained=pretrained)
        else:
            self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)        
        return x 

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        if not self.featurepred:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = self.myfc(dropout(x))
                else:
                    out += self.myfc(dropout(x))
            out /= len(self.dropouts)
        else:
            out = x        
        return out


class Resnest_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, featurepred=False):
        super(Resnest_Melanoma, self).__init__()
        self.featurepred = featurepred
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        if not self.featurepred: # Predict Softmax
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = self.myfc(dropout(x))
                else:
                    out += self.myfc(dropout(x))
            out /= len(self.dropouts)
        else:
            out = x
        return out


class Seresnext_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False, featurepred=False):
        super(Seresnext_Melanoma, self).__init__()
        self.featurepred = featurepred
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        if not self.featurepred:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = self.myfc(dropout(x))
                else:
                    out += self.myfc(dropout(x))
            out /= len(self.dropouts)
        else:
            out = x
        return out

########


class KWClassifier:
    """Classifier with 5 cv-folds of the same deepnet"""

    def __init__(self,
            model_n,            # str '01'..'18', or int 1 <= model_n <= 18
            weights_dir,        # if str of Path: load models for this directory
                                # if None: leave models uninitialized
            device = 'cpu',     # 'cpu' or 'cuda'
            version = 'final',
            featurepred=False
            ):
        # process parameters
        if isinstance(model_n, int):
            model_n = '{:02d}'.format(model_n)
        self.model_n = model_n
        self.device = torch.device(device)
        self.version = version
        self.featurepred = featurepred
        # obtain further parameters from modelspec
        spec = get_modelspec(self.model_n)
        self.kernel_type = spec.kernel_type
        self.data_folder = spec.data
        self.image_size = spec.imgsz
        # self.net_type
        for key, value in net_type_dict.items():
            if key in self.kernel_type:
                self.net_type = value
                break
        else:
            raise ValueError(f'{self.kernel_type}: unknown net_type')
        self.init_lr = spec.init_lr
        # self.out_dim
        try:
            self.out_dim = int(self.kernel_type[0])
        except Exception:
            raise ValueError(f'{self.kernel_type}: unknown out_dim')
        # self.default_n_epochs
        if '_15ep' in self.kernel_type:
            self.default_n_epochs = 15
        elif '_18ep' in self.kernel_type:
            self.default_n_epochs = 18
        else:
            raise ValueError(f'{self.kernel_type}: unknown default_n_epochs')
        self.use_meta = ('meta' in self.kernel_type)
        self.n_meta_features = 14 if self.use_meta else 0
        self.n_meta_dim = (512, 128)
        if '_meta128_32_' in self.kernel_type:
            self.n_meta_dim = (128, 32)
        if '_feature_' in self.kernel_type:
            feature_name = self.kernel_type.partition('_feature_')[2]
            self.diagnosis2idx = derm7pt_support.diagnosis2idx[feature_name]
            self.mel_idx = np.array(list(derm7pt_support.is_positive[feature_name]), dtype=np.uint32)
        else:
            if self.out_dim == 9:
                self.diagnosis2idx = {
                        'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'SCC': 4, 'VASC': 5,
                        'MEL': 6, 'NV': 7, 'UNK': 8,
                        }
            elif self.out_dim == 4:
                self.diagnosis2idx = {
                        'AK': 3, 'BCC': 3, 'BKL': 0, 'DF': 3, 'SCC': 3, 'VASC': 3,
                        'MEL': 1, 'NV': 2, 'UNK': 3,
                        }
            self.mel_idx = np.array([self.diagnosis2idx['MEL']], dtype=np.uint32)

        self.models = None
        if weights_dir is not None:
            self.build_and_load(weights_dir,version=self.version)

    def _build_model(self, pretrained=False):
        """build single model, return it"""
        if self.net_type == 'resnest101':
            ModelClass = Resnest_Melanoma
        elif self.net_type == 'seresnext101':
            ModelClass = Seresnext_Melanoma
        elif 'efficientnet' in self.net_type:
            ModelClass = Effnet_Melanoma
        else:
            raise NotImplementedError()

        model = ModelClass(
            self.net_type,
            n_meta_features = self.n_meta_features,
            n_meta_dim = self.n_meta_dim,
            out_dim = self.out_dim,
            pretrained = pretrained,
            featurepred = self.featurepred
        )
        return model

    def build_and_load(self, weights_dir, version='final'):
        """build 5 models and load weights"""
        self.models = []
        for fold in range(5):
            model = self._build_model()
            model.to(self.device)
            path = (Path(weights_dir) 
                    / f'weights_m{self.model_n}_f{fold}_{version}.pth')
            if not path.exists():
                print(f'skip {path}')
                self.models.append(None)
                continue
            print(f'load {path} ...')
            try:    # single GPU
                model.load_state_dict(torch.load(path, map_location=self.device), strict=True)
                #model.load_state_dict(torch.load(path), strict=True)
            except: # multi GPU
                state_dict = torch.load(path)
                # strip leading "module." from keys
                state_dict = { k_[7:] if k_.startswith('module.') else k_: 
                        state_dict[k_] for k_ in state_dict.keys() }
                model.load_state_dict(state_dict, strict=True)
            self.models.append(model)

    def predict_softmax(self, list_of_images:List[np.ndarray],
            list_of_meta:List[Dict],
            batch_size=16,
            n_workers=1,
            ) -> np.ndarray:
        """predict on a list of images+meta
        
        list_of_meta contains dicts per image, eg:
            {'sex': 'male', 'age':45} or {'sex':None, 'age':None}
        """
        dataset = PredictDataset(list_of_images, list_of_meta,
                self.use_meta, self.image_size, self.data_folder)
        loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size,
                num_workers=n_workers,
                )
        softmax = self._predict(loader)[:, self.mel_idx].sum(1)
        return softmax

    @classmethod
    def get_trans(cls, img, I):
        """helper function: simple augmentation: flip/transpose"""
        if I >= 4:
            img = img.transpose(2, 3)
        if I % 4 == 0:
            return img
        elif I % 4 == 1:
            return img.flip(2)
        elif I % 4 == 2:
            return img.flip(3)
        elif I % 4 == 3:
            return img.flip(2).flip(3)

    def _predict(self,
            loader,
            models = None,  # default: self.models
            n_tta = 8, # number of flip/transpose trials for test time augment
            use_tqdm = False,
            ):
        """return prediction as np.array"""
        # params
        if models is None:
            models = self.models
        if use_tqdm:
            loader = tqdm(loader)
        #
        for model in models:
            try:
                model.eval()
            except Exception as e:
                print(e)
                raise RuntimeError('missing or illegal model')
        softmax = []
        with torch.no_grad():
            for data, meta in loader:
                data, meta = data.to(self.device), meta.to(self.device)
                #softmax_batch = torch.zeros((data.shape[0], self.out_dim))
                softmax_batch = None

                for model in models:
                    for I in range(n_tta):
                        logits = model(self.get_trans(data, I), meta)

                        if softmax_batch== None:
                            if self.featurepred:
                                softmax_batch = torch.zeros((data.shape[0], logits.softmax(1).shape[1]))                                
                            else:
                                softmax_batch = torch.zeros((data.shape[0], self.out_dim))
                            softmax_batch = softmax_batch.to(self.device)
                        softmax_batch += logits.softmax(1)                        

                softmax_batch /= n_tta
                softmax_batch /= len(models)
                softmax.append(softmax_batch.detach().cpu())
        return torch.cat(softmax).numpy()

# vim: set sw=4 sts=4 expandtab :
