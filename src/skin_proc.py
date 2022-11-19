#!/usr/bin/env python3

# scripts that create files, eg. training

# standard library
import datetime
import sys
import os
import pickle
import subprocess
from pathlib import Path
self_dir = Path(__file__).parent

# common numerical and scientific libraries
import numpy as np
from numpy import sqrt, round
import pandas as pd
import skimage.measure
import skimage.filters
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import train_test_split
#import h5py
import imageio
from tqdm import tqdm
import skimage

# other common libraries
import cv2  # opencv

# local
import scriptlib as scr
from probability import get_model_proba, ComputeProbability
from util import read_csv, write_csv
from skin_fig import Eval
sys.path.insert(0, str(self_dir / '../src'))
import derm7pt_support

# global parameters
default_data_root = 'data/kagglewinner'
default_n_workers = 8

diagnosis_map = {
        # isic2020 -> isic2019
        'atypical melanocytic proliferation': 'UNK',    # used in kagglewinner
        #'atypical melanocytic proliferation': 'NV',    # correct
        'cafe-au-lait macule': 'UNK',
        'lentigo NOS': 'BKL',
        'lichenoid keratosis': 'BKL',
        'melanoma': 'MEL',
        'nevus': 'NV',
        'seborrheic keratosis': 'BKL',
        'solar lentigo': 'BKL',
        'unknown': 'UNK',
        # identity for isic2019
        'AK': 'AK',
        'BCC': 'BCC',
        'BKL': 'BKL',
        'DF': 'DF',
        'MEL': 'MEL',
        'NV': 'NV',
        'SCC': 'SCC',
        'VASC': 'VASC',
        'UNK': 'UNK',
        # ham10000: little change (AKIEC->AK; SCC,UNK missing)
        'AKIEC': 'AK',
        # derm7pt
        'basal cell carcinoma': 'BCC',
        'blue nevus': 'NV',
        'clark nevus': 'NV',
        'combined nevus': 'NV',
        'congenital nevus': 'NV',
        'dermal nevus': 'NV',
        'dermatofibroma': 'UNK',    # TODO correct?
        'lentigo': 'BKL',
        #'melanoma': 'MEL',
        'melanoma (0.76 to 1.5 mm)': 'MEL',
        'melanoma (in situ)': 'MEL',
        'melanoma (less than 0.76 mm)': 'MEL',
        'melanoma (more than 1.5 mm)': 'MEL',
        'melanoma metastasis': 'MEL',
        'melanosis': 'BKL',
        'miscellaneous': 'UNK',
        'recurrent nevus': 'NV',
        'reed or spitz nevus': 'NV',
        #'seborrheic keratosis': 'BKL',
        'vascular lesion': 'VASC',
        }

anatom_site_map = {
        # derm7pt
        'abdomen': 'anterior torso',
        'acral': 'palms/soles',
        'back': 'posterior torso',
        'buttocks': 'posterior torso',
        'chest': 'anterior torso',
        'genital areas': 'oral/genital',
        'head neck': 'head/neck',
        'lower limbs': 'lower extremity',
        'upper limbs': 'upper extremity',
        # ham10000
        'scalp': 'head/neck',
        'trunk': 'torso',
        #abdomen
        'genital': 'oral/genital',
        'ear': 'head/neck',
        #back
        'hand': 'upper extremity',
        #acral
        'unknown': None,
        'foot': 'lower extremity',
        'face': 'head/neck',
        'neck': 'head/neck',
        #chest
        #lower extremity
        #upper extremity
        }


kagglewinner_tfrecord_to_fold = {   # general case (not newfold or oldfold)
    2:0, 4:0, 5:0,
    1:1, 10:1, 13:1,
    0:2, 9:2, 12:2,
    3:3, 8:3, 11:3,
    6:4, 7:4, 14:4,
}

class CreateDSD(scr.SubCommand):
    name = 'create-dsd'
    description = 'create dataset descriptor csv file from a known dataset'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--dataset',
                required=True, choices=[
                    'directory-based',
                    'kagglewinner-train-stdfold',
                    'kagglewinner-train-oldfold',
                    'kagglewinner-train-newfold',
                    'kagglewinner-test',
                    'ham10000',
                    'ham10000-rosendahl',
                    'ham10000-vidir_modern',
                    'derm7pt-derm',
                    'derm7pt-macro',
                    'isic2018-task2',
                    'isic2019',
                    'ph2',
                    'pad-ufes',
                    'mednode',
                    ],
                help='dataset type')
        parser.add_argument('--directory-categories',
                help='specify nonmel_cat,mel_cat for --dataset=directory-based')
        parser.add_argument('--kagglewinner-datasize',
                help='specify 512, 768 or 1024 for --dataset=kagglewinner-*')
        parser.add_argument('--derm7pt-feature',
                choices = derm7pt_support.diagnosis2idx.keys(),
                help='choose feature target. Default: melanoma')
        parser.add_argument('--dataset-root',
                required=True,
                help='root directory of dataset')
        parser.add_argument('--sort',
                action='store_true',
                help='sort (for v02 compatibility): target=0 first, target=1 '
                    'last, alphabetical in name within class')
        parser.add_argument('out_path_csv',
                help='path of output csv file')

    def run(self):
        p = self.p
        # process parameters
        out_path_csv = Path(self.p.out_path_csv)
        out_path_csv.parent.mkdir(parents=True, exist_ok=True)
        #
        df = getattr(self, f'generate_{p.dataset.replace("-", "_")}')()
        if p.sort:
            df.sort_values(by=['target', 'name'], inplace=True)
        write_csv(out_path_csv, df, index=False)

    def generate_directory_based(self):
        # process parameters
        dataset_root = Path(self.p.dataset_root)
        # catdirs
        if self.p.directory_categories is None:
            catdirs = sorted(list(dataset_root.glob('*')))
            assert len(catdirs) == 2, f'expected 2 categories, found {len(catdirs)}'
        else:
            tmp = self.p.directory_categories.split(',')
            catdirs = [dataset_root / p_ for p_ in tmp]
        print(f'Negative (non-melanoma) category: {catdirs[0].name}')
        print(f'Positive (melanoma) category: {catdirs[1].name}')
        dsd_names = []
        dsd_paths = []
        dsd_targets = []
        for cat_i, catdir in enumerate(catdirs):
            paths = sorted(list(catdir.glob('*')))
            for path in paths:
                relpath = path.relative_to(dataset_root)
                name = f'{catdir.name}_{relpath.stem}'
                dsd_names.append(name)
                dsd_paths.append(relpath)
                dsd_targets.append(cat_i)
        if len(set(dsd_names)) != len(dsd_names):
            raise RuntimeError('duplicate file name(s): ' + 
                    f'{[x_ for x_ in dsd_names if dsd_names.count(x_) > 1]}')
        return pd.DataFrame({
            'name': dsd_names,
            'path': dsd_paths,
            'age': None,
            'sex': None,
            'anatom_site': None,
            'target': dsd_targets,
            })


    def generate_kagglewinner_train_stdfold(self):
        return self._generate_kagglewinner(role='train', foldsel='std')

    def generate_kagglewinner_train_oldfold(self):
        return self._generate_kagglewinner(role='train', foldsel='old')

    def generate_kagglewinner_train_newfold(self):
        return self._generate_kagglewinner(role='train', foldsel='new')

    def generate_kagglewinner_test(self):
        return self._generate_kagglewinner(role='test')

    def _generate_kagglewinner(self, role, foldsel=None):
        # process parameters
        dataset_root = Path(self.p.dataset_root)
        #
        tmp = self.p.kagglewinner_datasize
        dsize = f'{tmp}x{tmp}'
        outdf_all = pd.DataFrame()
        if role == 'train':
            subdirs = [f'jpeg-melanoma-{dsize}', f'jpeg-isic2019-{dsize}']
        elif role == 'test':
            subdirs = [f'jpeg-melanoma-{dsize}',]
        for subdir in subdirs:
            csv_path = dataset_root / subdir / f'{role}.csv'
            df = pd.read_csv(csv_path)
            # drop tfrecord < 0 immediately from both subdirs
            df = df[df['tfrecord'] >= 0].reset_index(drop=True)
            # replace patient_id==-1 with nan
            df.loc[df['patient_id'] == -1, 'patient_id'] = np.nan
            outdf = pd.DataFrame({
                'name': df['image_name'],
                'path': f'{subdir}/{role}/' + df['image_name'] + '.jpg',
                'age': df['age_approx'],
                'sex': df['sex'],
                'anatom_site': df['anatom_site_general_challenge'],
                'patient_id': df['patient_id'],
                })
            if role == 'train':
                outdf['target'] = df['target']
                outdf['diagnosis'] = df['diagnosis'].map(
                        diagnosis_map).fillna('UNK')
                if 'melanoma' in subdir:
                    # isic2020
                    if foldsel == 'new':
                        foldmap = {
                                8:0, 5:0, 11:0,
                                7:1, 0:1, 6:1,
                                10:2, 12:2, 13:2,
                                9:3, 1:3, 3:3,
                                14:4, 2:4, 4:4,
                                }
                    elif foldsel == 'old':
                        foldmap = {i_: i_ % 5 for i_ in range(15)}
                    elif foldsel == 'std':
                        foldmap = {
                                2:0, 4:0, 5:0,
                                1:1, 10:1, 13:1,
                                0:2, 9:2, 12:2,
                                3:3, 8:3, 11:3,
                                6:4, 7:4, 14:4,
                                }
                    outdf['fold'] = df['tfrecord'].map(foldmap)
                else:
                    # isic2019+2018
                    if foldsel == 'new':
                        foldmap = {
                                8:0, 5:0, 11:0,
                                7:1, 0:1, 6:1,
                                10:2, 12:2, 13:2,
                                9:3, 1:3, 3:3,
                                14:4, 2:4, 4:4,
                                }
                        outdf['fold'] = (df['tfrecord'] % 15).map(foldmap)
                    else:
                        outdf['fold'] = df['tfrecord'] % 5
            outdf_all = pd.concat([outdf_all, outdf])
        return outdf_all

    def generate_ham10000(self):
        return self._generate_ham10000()

    def generate_ham10000_rosendahl(self):
        return self._generate_ham10000(dataset='rosendahl')

    def generate_ham10000_vidir_modern(self):
        return self._generate_ham10000(dataset='vidir_modern')

    def _generate_ham10000(self, dataset=None):
        # set and process parameters
        n_folds = 5
        random_seed = 42
        dataset_root = Path(self.p.dataset_root)
        #
        csv_path = (dataset_root / 'HAM10000_metadata.csv')
        df = pd.read_csv(csv_path)
        if dataset is not None:
            df = df[df['dataset'] == dataset]
        lesion_ids = sorted(list(set(df['lesion_id'])))
        image_ids = []
        age_list = []
        sex_list = []
        anatom_site_list = []
        target_list = []
        diagnosis_list = []
        for lesion_id in lesion_ids:
            sel = df[df['lesion_id'] == lesion_id]
            sel = sel.sort_values(by='image_id')
            entry = sel.iloc[-1]
            image_ids.append(entry['image_id'])
            age_list.append(entry['age'])
            sex_list.append(entry['sex'])
            anatom_site = anatom_site_map.get(entry['localization'],
                    entry['localization'])
            anatom_site_list.append(anatom_site)
            diagnosis = diagnosis_map[entry['dx'].upper()]
            diagnosis_list.append(diagnosis)
            target_list.append(int(diagnosis == 'MEL'))
            #print(lesion_id, diagnosis, target_list[-1])
            #print(sel)
        outdf = pd.DataFrame({
            'name': image_ids,
            'path': [f'images/{n_}.jpg' for n_ in image_ids],
            'age': age_list,
            'sex': sex_list,
            'anatom_site': anatom_site_list,
            'target': target_list,
            'diagnosis': diagnosis_list,
            'fold': self.calc_fold(stratify=diagnosis_list),
            })
        return outdf

    def generate_isic2018_task2(self):
        # set and process parameters
        n_folds = 5
        dataset_root = Path(self.p.dataset_root)
        train_dir = dataset_root / 'ISIC2018_Task1-2_Training_Input'
        #
        names = sorted([p_.stem for p_ in train_dir.glob('*.jpg')])
        outdf = pd.DataFrame({
            'name': names,
            'path': [f'ISIC2018_Task1-2_Training_Input/{n_}.jpg'
                for n_ in names],
            'fold': self.calc_fold(stratify=np.zeros(len(names))),
            })
        return outdf

    def generate_isic2019(self):
        # set and process parameters
        n_folds = 5
        random_seed = 42
        dataset_root = Path(self.p.dataset_root)
        truth_path = dataset_root / 'ISIC_2019_Training_GroundTruth.csv'
        meta_path  = dataset_root / 'ISIC_2019_Training_Metadata.csv'
        #
        df_truth = pd.read_csv(truth_path)
        df_meta = pd.read_csv(meta_path)
        assert (df_truth['image'] == df_meta['image']).sum() == len(df_truth)
        categories = df_truth.columns.values[1:]
        diagnosis = categories[df_truth[categories].values.argmax(axis=1)]
        outdf = pd.DataFrame({
            'name': df_meta['image'],
            'path': [f'ISIC_2019_Training_Input/{n_}.jpg'
                for n_ in df_meta['image']],
            'age': df_meta['age_approx'],
            'sex': df_meta['sex'],
            'anatom_site': df_meta['anatom_site_general'],
            'lesion_id': df_meta['lesion_id'],
            'target': (diagnosis == 'MEL').astype(int),
            'diagnosis': diagnosis,
            })
        self.set_fold_lesionid(outdf)
        self.test_fold_lesionid(outdf)
        return outdf


    def generate_derm7pt_derm(self):
        return self._generate_derm7pt(mode='derm')

    def generate_derm7pt_macro(self):
        return self._generate_derm7pt(mode='macro')

    def _generate_derm7pt(self, mode):
        # process parameters
        dataset_root = Path(self.p.dataset_root)
        feature = self.p.derm7pt_feature
        #
        mode_col = {
                'derm': 'derm',
                'macro': 'clinic',
                }[mode]
        in_path = dataset_root / 'meta/meta.csv'
        df = pd.read_csv(in_path)
        outdf = pd.DataFrame({
            'name': [f'{mode}-{cn_:04d}' for cn_ in df['case_num']],
            'path': 'images/' + df[mode_col],
            'age': None,
            'sex': df['sex'],
            'anatom_site': df['location'].map(anatom_site_map),
            })
        if feature is None:
            outdf['diagnosis'] = df['diagnosis'].map(diagnosis_map)
            outdf['target'] = (outdf['diagnosis'] == 'MEL').astype(int)
        else:
            outdf['diagnosis'] = df[feature]
            outdf['target'] = [
                    int(derm7pt_support.diagnosis2idx[feature][val_]
                        in derm7pt_support.is_positive[feature])
                    for val_ in outdf['diagnosis']]
        outdf['fold'] = self.calc_fold(stratify=outdf['diagnosis'].values)
        return outdf

    def generate_ph2(self):
        # process parameters
        dataset_root = Path(self.p.dataset_root)
        #
        name = []
        histological_diagnosis = []
        clinical_diagnosis = []
        asymmetry = []
        pigment_network = []
        dots_globules = []
        streaks = []
        regression_areas = []
        blue_whitish_veil = []
        colors = []
        for line in open(dataset_root / 'PH2_dataset.txt'):
            words = line.split('|')
            if len(words) < 16 or words[2].strip() == 'Name':
                continue
            name1 = words[2].strip()
            name.append(name1)
            histological_diagnosis.append(words[4].strip())
            clinical_diagnosis.append(words[6].strip())
            asymmetry.append(words[8].strip())
            pigment_network.append(words[9].strip())
            dots_globules.append(words[10].strip())
            streaks.append(words[11].strip())
            regression_areas.append(words[12].strip())
            blue_whitish_veil.append(words[13].strip())
            colors.append(words[15].strip().replace(' ', ''))
        outdf = pd.DataFrame({
            'name': name,
            'path': [f'PH2 Dataset images/{n_}/{n_}_Dermoscopic_Image/{n_}.bmp'
                for n_ in name],
            'mask_path': [f'PH2 Dataset images/{n_}/{n_}_lesion/{n_}_lesion.bmp'
                for n_ in name],
            'target': [int(cd_ == '2') for cd_ in clinical_diagnosis],
            'age': None,
            'sex': None,
            'ph2_histological_diagnosis': histological_diagnosis,
            'ph2_clinical_diagnosis': clinical_diagnosis,
            'ph2_asymmetry': asymmetry,
            'ph2_pigment_network': pigment_network,
            'ph2_dots_globules': dots_globules,
            'ph2_streaks': streaks,
            'ph2_regression_areas': regression_areas,
            'ph2_blue_whitish_veil': blue_whitish_veil,
            'ph2_colors': colors,
            })
        outdf['fold'] = self.calc_fold(
                stratify=outdf['ph2_clinical_diagnosis'].values)
        return outdf

    def generate_pad_ufes(self):
        # set and process parameters
        n_folds = 5
        random_seed = 42
        dataset_root = Path(self.p.dataset_root)
        meta_path  = dataset_root / 'metadata.csv'
        #
        df_meta = pd.read_csv(meta_path)
        sex = df_meta['gender'].copy()
        sex[sex == 'MALE'] = 'male'
        sex[sex == 'FEMALE'] = 'female'
        sex[sex.isna()] = None
        outdf = pd.DataFrame({
            'name': [n_.strip('.png') for n_ in df_meta['img_id']],
            'path': [f'images/{n_}' for n_ in df_meta['img_id']],
            'sex': sex,
            'diagnosis': df_meta['diagnostic'],
            })
        for col in 'patient_id,lesion_id,smoke,drink,background_father,background_mother,age,pesticide,gender,skin_cancer_history,cancer_history,has_piped_water,has_sewage_system,fitspatrick,region,diameter_1,diameter_2,itch,grew,hurt,changed,bleed,elevation,biopsed'.split(','):
            outdf[col] = df_meta[col]
        self.set_fold_lesionid(outdf)
        self.test_fold_lesionid(outdf)
        return outdf

    @classmethod
    def calc_fold(cls, stratify):
        # parameters
        n_folds = 5
        random_seed = 42
        #
        cv = StratifiedKFold(n_folds, shuffle=True, random_state=random_seed)
        fold_arr = np.zeros(len(stratify), dtype=int)
        fold_arr[:] = -1
        for fold, (train, test) in enumerate(
                cv.split(np.zeros(len(stratify)), stratify)):
            fold_arr[test] = fold
        assert (fold_arr < 0).sum() == 0
        return fold_arr

    @classmethod
    def set_fold_lesionid(cls, df):
        """sets 'fold' columns in df, based on lesion_id and diagnosis
        Images with same lesion_id do not cross fold boundaries.
        Side effect: column 'unique_lesion_id' is set.
        """
        # parameters
        n_folds = 5
        random_seed = 42
        # init random
        rng = np.random.default_rng(seed=random_seed)
        # set unique lesion_id when missing
        df['unique_lesion_id'] = df['lesion_id']
        for i, row in df.iterrows():
            if not isinstance(row['unique_lesion_id'], str):
                df.loc[i, 'unique_lesion_id'] = f'UNIQUE_{i:05d}'
        df.set_index('name', drop=False, inplace=True)
        # lesions
        lesion_id_list = sorted(list(set(df['unique_lesion_id'])))
        les = pd.DataFrame({
            'num': [(df['unique_lesion_id'] == id_).sum()
                for id_ in lesion_id_list],
            'diagnosis': [df[df['unique_lesion_id'] == id_].iloc[0]['diagnosis']
                for id_ in lesion_id_list],
            }, index=lesion_id_list)
        les = les.sample(frac=1, random_state=random_seed) #controlled randomize
        les.sort_values(by='num', kind='mergesort', ascending=False,
                inplace=True)   # lesions with highest img count first
        # distribute lesion-grouped images into folds
        df['fold'] = -1
        for diag in set(les['diagnosis']):
            les_diag = les[les['diagnosis'] == diag]
            bins = [[] for i_ in range(n_folds)]
            for lesion_id, row in les_diag.iterrows():
                assert row.diagnosis == diag
                # select the bin / one of the bins with lowest image count
                counts = np.array([len(b_) for b_ in bins])
                min_indices = np.nonzero(counts == counts.min())[0]
                selected_idx = rng.choice(min_indices)
                # deposit images
                names = df[df['unique_lesion_id'] == lesion_id]['name']
                bins[selected_idx] += list(names)
            # set fold
            for i in range(n_folds):
                df.loc[bins[i], 'fold'] = i
        assert (df['fold'] < 0).sum() == 0

    @classmethod
    def test_fold_lesionid(cls, df):
        for lesion_id in set(df['lesion_id']):
            if not isinstance(lesion_id, str):
                continue
            samples = df[df['lesion_id'] == lesion_id]
            assert len(set(samples['fold'])) == 1, \
                    f'lesion "{lesion_id}" in multiple folds: {samples}'


class Train(scr.SubCommand):
    name = 'train'

    @classmethod
    def add_arguments(self, parser):

        parser.add_argument('--model-n',
                type=str, required=True,
                help='2-character model number')
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root [default: %(default)s]')
        parser.add_argument('--n-epochs',
                type=int, 
                help='number of epochs [default: 15 or 18, defined in model]')
        parser.add_argument('--batch-size',
                type=int, required=True,
                help='batch size')
        parser.add_argument('--weighted',
                action='store_true',
                help='class weighted training')
        parser.add_argument('--n-workers',
                type=int, default=default_n_workers,
                help='number of workers [default: %(default)d]')
        parser.add_argument('--folds',
                type=str, default='0,1,2,3,4',
                help='folds to train [default: %(default)s]')
        parser.add_argument('--use-amp',
                type=scr.str2bool, default=True,
                help='use AMP [default: %(default)s]')
        parser.add_argument('--use-aug',
                type=scr.str2bool, default=True,
                help='use aumgmentation [default: %(default)s]')
        parser.add_argument('--seed',
                type=int, default=0,
                help='rng seed [default: %(default)d]')
        parser.add_argument('--lr',
                type=float, 
                help='initial lr (default: model defined)')
        parser.add_argument('--fix-lr',
                action='store_true',
                help='use fixed lr. Default: cosine annealing')
        parser.add_argument('--weight-decay',
                type=float, default=0.,
                help='weight decay for optimizer')
        parser.add_argument('--sam',
                action='store_true',
                default=False,
                help='turns on Sharpness Aware Minimization (SAM) Default: %(default)s')
        parser.add_argument('--sam-rho',
                type=float, default=0.05,
                help='Rho of SAM if it is turned on Default: %(default)s')
        parser.add_argument('--adaptive-sam',
                action='store_true',
                default=False,
                help='Turns on adaptiveness for SAM Default: %(default)s')

        parser.add_argument('out_dir',
                help='output model and log directory')


    def run(self):
        from tkw_classifier import TrainableKWClassifier
        p = self.p
        folds = [int(i_) for i_ in p.folds.split(',')]

        kwc = TrainableKWClassifier(p.model_n, None, device='cuda')
        if p.lr is not None:
            kwc.init_lr = p.lr
        dsd = read_csv(p.dsd)
        kwc.train(p.out_dir, dsd, p.data_root, p.batch_size,
                is_weighted = p.weighted,
                n_epochs = p.n_epochs,
                n_workers = p.n_workers,
                folds = folds,
                use_amp = p.use_amp,
                use_aug = p.use_aug,
                seed = p.seed,
                fix_lr = p.fix_lr,
                weight_decay = p.weight_decay,
                sam = p.sam,
                sam_rho = p.sam_rho,
                sam_adaptive = p.adaptive_sam
                )
        print('elapsed:', self.elapsed(True))


class Predict(scr.SubCommand):
    name = 'predict'
    description = 'predict on a single kw model'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-n',
                type=str, required=True,
                help='2-character model number')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root')
        parser.add_argument('--oof',
                action='store_true',
                help='do out-of-fold prediction on cv-trained models')
        parser.add_argument('--folds',
                type=str, default='0,1,2,3,4',
                help='folds used in prediction [default: %(default)s]. Unused when --oof.')
        parser.add_argument('--batch-size',
                type=int, required=True,
                help='batch size')
        parser.add_argument('--n-workers',
                type=int, default=default_n_workers,
                help='number of workers [default: %(default)d]')
        parser.add_argument('--debug',
                action='store_true',
                help='debugging: restrict data size')
        parser.add_argument('--feature',
                action='store_true',
                help='predict last layer feature values')
        parser.add_argument('--version',
                type=str, default='best',
                help='model version "final/best"')
        parser.add_argument('--cpu',
                action='store_true',
                help='cpu only [default: cuda]')
        parser.add_argument('out_dir',
                help='output model and log directory')

    def run(self):
        from tkw_classifier import TrainableKWClassifier
        p = self.p
        
        device = 'cpu' if p.cpu else 'cuda'
        kwc = TrainableKWClassifier(p.model_n, weights_dir=p.model_dir,
                device=device, version=p.version, featurepred=p.feature)
        dsd = read_csv(p.dsd)
        
        if p.oof:
            if 'fold' not in dsd.columns:
                raise RuntimeError('fold data missing for --oof')
            kwc.predict_oof(dsd, p.data_root, p.batch_size, p.n_workers,
                    out_dir=p.out_dir,
                    )
        else:
            models = [kwc.models[int(i_)] for i_ in p.folds.split(',')]
            kwc.predict(dsd, p.data_root, p.batch_size, p.n_workers,
                    models=models,
                    out_dir=p.out_dir, 
                    )
        print('elapsed:', self.elapsed(True))

@scr.skip
class CalcAvg(scr.SubCommand):
    name = 'calc-avg'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--input',
                nargs='*',
                help='input csv files')
        parser.add_argument('out_path',
                help='output csv file')


    def run(self):
        p = self.p
        out_path = Path(p.out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        #
        softmaxes = []
        for i, input_csv in enumerate(p.input):
            sm = read_csv(input_csv, index_col=0)
            softmaxes.append(sm)
        avg = np.array([df_['pred'] for df_ in softmaxes]).mean(axis=0)
        sm_all = pd.DataFrame({
            'name': softmaxes[0].index,
            'pred': avg,
            'true': softmaxes[0]['true'],
            })
        write_csv(out_path, sm_all, index=False)


@scr.skip
class TrainCM(scr.SubCommand):
    name = 'train-cm'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--input',
                nargs='*',
                help='input csv files')
        parser.add_argument('--n-epochs',
                type=int, default=100,
                help='number of epochs. Default: %(default)d')
        parser.add_argument('--batch-size',
                type=int, default=64,
                help='batch size. Default: %(default)d')
        parser.add_argument('--dropout',
                type=float, default=0.1,
                help='dropout. Default: %(default)g')
        parser.add_argument('--noise',
                type=float, default=0.0,
                help='noise. Default: %(default)g')
        parser.add_argument('--init-to-avg',
                action='store_true',
                help='initialize weights to calculate noisy average')
        parser.add_argument('--lr',
                type=float, default=1e-4,
                help='learning rate')
        parser.add_argument('--seed',
                type=int, default=0,
                help='rng seed [default: %(default)d]')
        parser.add_argument('--device',
                default='cpu',
                help='cpu or cuda. Default: %(default)s')
        parser.add_argument('out_dir',
                help='output model and log directory')

    def run(self):
        from tkw_committee import (TrainableKWCommitteeMachine, load_split,
                load_inputs)
        p = self.p
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        #
        cm = TrainableKWCommitteeMachine(weights_dir=None, device=p.device)
        dsd = read_csv(p.dsd)
        data = load_inputs(p.input, is_v04=0*True)
        cm.train_predict(
                out_dir,
                dsd,
                data,
                p.lr,
                p.n_epochs,
                p.batch_size,
                p.dropout,
                p.noise,
                init_to_avg = p.init_to_avg,
                seed = p.seed,
                )
        print('elapsed:', self.elapsed(True))


class PredictCM(scr.SubCommand):
    name = 'predict-cm'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--noise',
                type=float, default=0.0,
                help='noise. Default: %(default)g')
        parser.add_argument('--input',
                nargs='*',
                help='input h5 (v02 models) or csv (kw models) files')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('out_dir',
                help='output directory')

    def run(self):
        from tkw_committee import TrainableKWCommitteeMachine, load_inputs
        p = self.p
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        #
        dsd = read_csv(p.dsd)
        x = load_inputs(p.input, is_v04=0*True)
        print('input_shape:', x.shape)
        yt = np.array(dsd['target'])
        cm = TrainableKWCommitteeMachine(weights_dir=p.model_dir)
        df = pd.DataFrame({
            'name': dsd['name'],
            'true': yt,
            'pred': cm.predict(x, p.noise),
            })
        path = out_dir / 'softmax_cm.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(path, df, index=False)

class CalibratePredictProba(scr.SubCommand):
    name = 'calibrate-predict-proba'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data',
                required=True,
                help='path to result csv file (name, pred, true)')
        parser.add_argument('--beta',
                type=float, default=1,
                help='beta parameter for softmax rescaling, eg. 0.3')
        parser.add_argument('--n-bins',
                type=int, default=30,
                help='number of bins')
        parser.add_argument('--out-model-path',
                required=True,
                help='output model path (h5)')
        parser.add_argument('--plot',
                action='store_true',
                help='plot calibration curve')
        parser.add_argument('out_path',
                help='path to output csv file (name, pred=proba, true)')

    def run(self):
        p = self.p
        df = read_csv(p.data)
        model_proba, misc = get_model_proba(df['true'], df['pred'], p.beta,
                p.n_bins, return_misc=True)
        #pickle.dump((model_proba, misc), open('qtest.pickle', 'wb'))
        model_proba.save(p.out_model_path)
        yproba = model_proba.predict(df['pred'])
        df['pred'] = yproba
        write_csv(p.out_path, df, index=False)
        if p.plot:
            from probability import sigma, inv_sigma
            from matplotlib.pyplot import (plot, legend, show, xlim, xlabel,
                    ylabel, bar, savefig)
            beta = 1. / model_proba.k
            _, bin_proba = misc
            n_bins = len(bin_proba)
            s = np.linspace(0, 1, 1000)
            s = np.concatenate([
                    np.linspace(0, 0.001, 1000),
                    np.linspace(0.001, 0.999, 10000),
                    np.linspace( 0.999, 1, 1000),
                    ])
            ss = sigma(inv_sigma(s, 1, 0), beta, 0)
            bar(np.linspace(0, 1, n_bins), bin_proba, width=1/(n_bins + 3),
                    label='positive fraction in bin', color='red')
            plot(ss, model_proba.predict(s), '-', label='fitted sigmoid')
            legend()
            #xlim(0, 1)
            xlabel('$s\'$')
            ylabel('positive fraction')
            savefig('calibproba.pdf')
            show()
            


class PredictProba(scr.SubCommand):
    name = 'predict-proba'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data',
                required=True,
                help='path to result csv file (name, pred, [true])')
        parser.add_argument('--model-path',
                required=True,
                help='model path')
        parser.add_argument('out_path',
                help='path to output csv file (name, pred=proba, [true])')

    def run(self):
        p = self.p
        df = read_csv(p.data)
        model_proba = ComputeProbability(p.model_path)
        yproba = model_proba.predict(df['pred'])
        df['pred'] = yproba # replace 'pred', keep other columns
        write_csv(p.out_path, df, index=False)


class PreprocFeatures(scr.SubCommand):
    name = 'preproc-features'
    
    @classmethod
    def add_arguments(cls, parser):        
        parser.add_argument('--data-root',
                required=True,
                help='root directory of dataset')
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('--out-img-size',
                default=256, type=int,
                help='Size of the output image')
        parser.add_argument('--part',
                default='0/1',
                help='sequence/total; eg. 3-way split: 0/3, 1/3, 2/3.')
        parser.add_argument('out_root',
                help='output root')

    def run(self):
        from segmenter import Segmenter
        from img_colorasym_proc import ImageColorAsym
        from img_center_proc import ImageCenter
        from img_border_proc import ImageBorder
        p = self.p
        out_root = Path(p.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        pp_list = [    
           ('colorasym', ImageColorAsym()),
           ('center', ImageCenter(box_size=p.out_img_size)),           
           ('border', ImageBorder(dim=(p.out_img_size,p.out_img_size)))     
           ]
        seg_params = Segmenter.load_segmentation_model(model_dir=p.model_dir)
        seg = Segmenter(*seg_params)
        
        dsd = read_csv(p.dsd)
        dsd.sort_values(by=['name'], inplace=True)

        pathlist = dsd['path']
        
        sequence, total = [int(x_) for x_ in p.part.split('/', 1)]
        paths = np.array_split(pathlist,total)[sequence]
        
        for img_path in tqdm(paths):
            img = imageio.imread(Path(p.data_root) / img_path)
            filename = Path(img_path).with_suffix('').stem
            img_ext = Path(img_path).suffix
            mask = seg.run([img])[0]
            if mask.sum() == 0: # empty mask: replace by a circle
                rr, cc = skimage.draw.circle(
                            mask.shape[0]//2,
                            mask.shape[1]//2,
                            int(min(mask.shape)*0.4),
                            )
                mask[rr, cc] = True

            for feature, pp in pp_list:
                out_img_dir = os.path.dirname(img_path)
                out_dir = Path(out_root / f'{feature}' / f'{out_img_dir}')                
                out_dir.mkdir(parents=True,exist_ok=True)
                out_path = out_dir / f'{filename}{img_ext}'
                imag_out = (pp.run([img], [mask])[0]).astype(np.uint8)
                imageio.imwrite(out_path, imag_out)           

class PredictFeatures(scr.SubCommand):
    name = 'predict-features'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data-root',
                required=True,
                help='root directory of dataset')
        parser.add_argument('--preproc-data-root',
                required=True,
                help='root directory of preprocessed dataset')
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('--batch-size',
                type=int, default=16,
                help='batch size')
        parser.add_argument('out_dir',
                help='output directory')

    def run(self):
        p = self.p
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(p.model_dir)
        dsd = Path(p.dsd)

        self.log('Feature prediction started')
        features = {
            'border':'21',
            'center':'21',
            'colorasym':'21',
            'pigment_network':'33',
            'streaks':'34',
            'pigmentation':'35',
            'regression_structures':'36',
            'dots_and_globules':'37',
            'blue_whitish_veil':'32',
            'vascular_structures':'38',
            'baseline':'21',

            #'baseline1_ros_weighted':'21',
            #'baseline1_vm_weighted':'21',
            #'baseline4_ros_cv_weighted':'21',
            #'baseline4_vm_cv_weighted':'21',
            #'baseline6_ros_aug_cv_weighted':'21', 
            #'baseline6_vm_aug_cv_weighted':'21',
            #'baseline8_kag_aug_cv_weighted':'21',
            #'baseline_m21_b64':'21',
            #'baseline_m22_b29_456':'22',
            #'baseline_m23_b29_528':'23',
            }

        for feature in features.keys():
            self.log(f'Predicting {feature}')
            data_root = p.data_root if feature not in ['border','center','colorasym'] else Path(p.preproc_data_root) / feature
            folds = '0' if feature.startswith('baseline1') else '0,1,2,3,4'
            out_feature_dir = out_dir / feature
            out_feature_dir.mkdir(parents=True, exist_ok=True)
            Predict([
                f'--model-n={features[feature]}',
                f'--data-root={data_root}',
                f'--dsd={dsd}',
                f'--model-dir={model_dir / feature}',
                f'--batch-size={p.batch_size}',
                f'--version=best',
                f'--folds={folds}',
                f'{out_feature_dir}'
            ]).run()

        self.log(f'elapsed: {self.elapsed(True)}')

    def log(self, text):
        p = self.p
        print(text)
        with open(Path(p.out_dir) / 'predict-features.log', 'a') as f:
            stamp = datetime.datetime.now().isoformat(sep="_",
                    timespec="seconds")
            if text.startswith('\n'):
                print(file=f)
                text = text[1:]
            print(f'{stamp} {text}', file=f)

class Predict18(scr.SubCommand):
    name = 'predict-18'
    description = 'predict all kw models and combine'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--model-dir',
                #required=True,
                help='model directory')
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                #required=True,
                help='data root')
        parser.add_argument('--batch-size',
                type=int, #required=True,
                help='batch size')
        parser.add_argument('--n-workers',
                type=int, default=default_n_workers,
                help='number of workers [default: %(default)d]')
        parser.add_argument('--no-predict',
                action='store_true',
                help='do not predict kw models, instead read csv files')
        parser.add_argument('out_dir',
                help='output directory for softmax files')

    def run(self):
        from tkw_classifier import TrainableKWClassifier
        p = self.p

        out_dir = Path(p.out_dir)
        dsd = read_csv(p.dsd)
        softmaxes = []
        for model_i in range(1, 18+1):
            model_n = f'{model_i:02d}'
            if p.no_predict:
                sm = read_csv(out_dir / f'softmax_m{model_n}.csv',
                        index_col=0)
            else:
                kwc = TrainableKWClassifier(model_n, p.model_dir, device='cuda',
                        version='best')
                sm = kwc.predict(dsd, p.data_root, p.batch_size, p.n_workers,
                        out_dir=p.out_dir)
            softmaxes.append(sm)
        avg = np.array([df_['pred'] for df_ in softmaxes]).mean(axis=0)
        sm_all = pd.DataFrame({
            'name': softmaxes[0].index,
            'pred': avg,
            'true': softmaxes[0]['true'],
            })
        write_csv(out_dir / 'softmax_avg.csv', sm_all, index=False)

        if 0: # TODO
            model_proba = get_model_proba(dsd['target'], avg, 0.3, 30)
            print(f'fitted model_proba parameters: '
                    + f'{model_proba.k:f},'
                    + f'{model_proba.n_bins:d},'
                    + f'{model_proba.x0:f},'
                    + f'{model_proba.w:f},'
                    + '-1,-1,'
                    + f'{model_proba.opt_ba_thres:f}')
        print('elapsed:', self.elapsed(True))


class PredictEvalDir(scr.SubCommand):
    name = 'predict-eval-dir'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data-root',
                required=True,
                help='root directory of dataset, containing 0-other and '+
                    '1-melanoma subdirs')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('--stages',
                default='abcdef',
                help='a=create-dsd, b=predict-18, c=predict-5, d=predict-cm, '+
                    'e=predict-proba, f=print-result.  Default: %(default)s')
        parser.add_argument('--batch-size',
                type=int, default=16,
                help='batch size')
        parser.add_argument('out_dir',
                help='output directory')

    def run(self):
        p = self.p
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(p.model_dir)

        self.log(f'\nRunning stages={p.stages}')
        if 'a' in p.stages: # create-dsd
            self.log(f'######## stage a started')
            CreateDSD([
                f'--dataset=directory-based',
                f'--dataset-root={p.data_root}',
                f'{out_dir / "dsd.csv"}',
                ]).run()

        if 'b' in p.stages: # predict-18
            self.log(f'######## stage b started')
            Predict18([
                f'--model-dir={model_dir}',
                f'--dsd={out_dir / "dsd.csv"}',
                f'--data-root={p.data_root}',
                f'--batch-size={p.batch_size}',
                f'{out_dir}',
                ]).run()

        if 'c' in p.stages: # preproc-features
            self.log(f'######## stage c started')
            PreprocFeatures([
                f'--dsd={out_dir / "dsd.csv"}',
                f'--data-root={p.data_root}',
                f'--model-dir={model_dir}',
                f'{out_dir}',
                ]).run()
            self.log('predict-features')

        if 'd' in p.stages: # predict-features
            self.log(f'######## stage d started')
            PredictFeatures([
                f'--dsd={out_dir / "dsd.csv"}',
                f'--data-root={p.data_root}',
                f'--preproc-data-root={out_dir}',
                f'--model-dir={model_dir}',
                f'--batch-size={p.batch_size}',
                f'{out_dir}',
                ]).run()

        if 'e' in p.stages: # predict-cm
            self.log(f'######## stage e started')
            inputs = (sorted(
                    list(out_dir.glob('softmax_m[01]?.csv')) + 
                    list(out_dir.glob('baseline/softmax_m2?.csv')) + 
                    list(out_dir.glob('*/softmax_m3?.csv')),
                    key=lambda x_: x_.name)
                    + [ out_dir / 'border/softmax_m21.csv',
                        out_dir / 'center/softmax_m21.csv',  
                        out_dir / 'colorasym/softmax_m21.csv',  
                        ])
            #for i in inputs:
            #    print(i)
            inputs = [str(p_) for p_ in inputs]
            PredictCM([
                f'--dsd={out_dir / "dsd.csv"}',
                f'--input'] + inputs + [
                f'--model-dir={model_dir}',
                f'{out_dir}',
                ]).run()

        #if 'x' in p.stages: # predict-proba
        #    self.log(f'######## stage x started')
        #    PredictProba([
        #        f'--data={out_dir / "softmax_cm.csv"}',
        #        f'--model-path={model_dir / "model_proba.h5"}',
        #        f'{out_dir / "proba_cm.csv"}',
        #        ]).run()

        if 'f' in p.stages: # print-eval
            self.log(f'######## stage f started')
            Eval([
                #f'--proba-model={model_dir / "model_proba.h5"}',
                f'--output={out_dir / "result.txt"}',
                f'{out_dir / "softmax_cm.csv"}',
                ]).run()

        self.log(f'elapsed: {self.elapsed(True)}')

    def log(self, text):
        p = self.p
        print(text)
        with open(Path(p.out_dir) / 'predict-eval-dir.log', 'a') as f:
            stamp = datetime.datetime.now().isoformat(sep="_",
                    timespec="seconds")
            if text.startswith('\n'):
                print(file=f)
                text = text[1:]
            print(f'{stamp} {text}', file=f)





@scr.skip
class PredictFeatureDir(scr.SubCommand):
    name = 'predict-features-dir'

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data-root',
                required=True,
                help='root directory of dataset')
        parser.add_argument('--preproc-data-root',
                required=True,
                help='root directory of preprocessed dataset')
        parser.add_argument('--dsd',
                help='dataset descriptor csv file path, if not present it will be generated')
        parser.add_argument('--model-dir',
                required=True,
                help='model directory')
        parser.add_argument('--batch-size',
                type=int, default=16,
                help='batch size')
        parser.add_argument('out_dir',
                help='output directory')

    def run(self):
        p = self.p
        if not p.out_dir.startswith('/'):
            p.out_dir = f'/{p.out_dir}'
        out_dir = Path(p.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dir = Path(p.model_dir)

        if p.dsd is None: # create-dsd
            self.log(f"dsd parameter not specified, creating new dsd file in:\n{out_dir.absolute}")
            CreateDSD([
                f'--dataset=directory-based',
                f'--dataset-root={p.data_root}',
                f'{out_dir / "dsd.csv"}',
                ]).run()
            dsd = out_dir / "dsd.csv"
        else:
            dsd = Path(p.dsd)

        self.log('Feature prediction started')
        features = {
            'border':'21',
            'center':'21',
            'colorasym':'21',
            'pigment_network':'33',
            'streaks':'34',
            'pigmentation':'35',
            'regression_structures':'36',
            'dots_and_globules':'37',
            'blue_whitish_veil':'32',
            'vascular_structures':'38',
            'baseline1_ros_weighted':'21',
            'baseline1_vm_weighted':'21',
            'baseline4_ros_cv_weighted':'21',
            'baseline4_vm_cv_weighted':'21',
            'baseline6_ros_aug_cv_weighted':'21', 
            'baseline6_vm_aug_cv_weighted':'21',
            'baseline8_kag_aug_cv_weighted':'21',
            'baseline_m21_b64':'21',
            'baseline_m22_b29_456':'22',
            'baseline_m23_b29_528':'23',
            }

        for feature in features.keys():
            self.log(f'Predicting {feature}')
            data_root = p.data_root if feature not in ['border','center','colorasym'] else Path(p.preproc_data_root) / feature
            folds = '0' if feature.startswith('baseline1') else '0,1,2,3,4'
            out_feature_dir = out_dir / feature
            out_feature_dir.mkdir(parents=True, exist_ok=True)
            Predict([
                f'--model-n={features[feature]}',
                f'--data-root={data_root}',
                f'--dsd={dsd}',
                f'--model-dir={model_dir / feature}',
                f'--batch-size={p.batch_size}',
                f'--version=best',
                f'--folds={folds}',
                f'{out_feature_dir}'
            ]).run()

        self.log(f'elapsed: {self.elapsed(True)}')

    def log(self, text):
        p = self.p
        print(text)
        with open(Path(p.out_dir) / 'predict-eval-dir.log', 'a') as f:
            stamp = datetime.datetime.now().isoformat(sep="_",
                    timespec="seconds")
            if text.startswith('\n'):
                print(file=f)
                text = text[1:]
            print(f'{stamp} {text}', file=f)


class Mulsub(scr.SubCommand):
    name = 'mulsub'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--seed',
                type=int, default=42,
                help='rng seed. Default: %(default)d')
        parser.add_argument('--model-n',
                type=int, default=41,
                help='model number, eg. 41')
        parser.add_argument('--batch-size',
                type=int, default=14,
                help='minibatch size. Default: %(default)d')
        parser.add_argument('--pipe',
                default='cuda',
                help='which pipe to use with btclient submit')
        parser.add_argument('--dry-run',
                action='store_true',
                help='do not submit, just print cmd')
        parser.add_argument('version',
                choices=['cv', 'cv_wt', 'cv_wt_aug',] +
                    [f'cv_wt_aug_sam{i}' for i in [1,2,3,4]] +
                    ['cv_wt_aug_wd1e-5', 'cv_wt_aug_wd1e-4', 'cv_wt_aug_wd1e-3',
                        'cv_wt_wd1e-5', 'cv_wt_wd3e-5', 'cv_wt_wd1e-3',
                        'cv_wt_adamw', 'cv_wt_aug_adamw',
                        'cv_wt_aug_mod1', 'cv_wt_aug_mod2', 'cv_wt_aug_mod3',
                        'cv_wt_aug_mod4',],
                help='which bl version to run')
        parser.add_argument('stage',
                choices=('train', 'pred', 'fit-proba', 'pred-proba'),
                help='which stage to run')

    def execute(self, cmd): 
        p = self.p
        if p.dry_run:
            print(cmd)
        else:
            subprocess.run(['btclient.py', 'submit', p.pipe, cmd])

    def run(self):
        p = self.p
        dsd_map = {
                'ros': 'v05/dsd_ros.csv',
                'vm': 'v05/dsd_vm.csv',
                'd7d': 'v05/dsd_d7d.csv',
                }
        data_root_map = {
                'ros': 'data/ham10000',
                'vm': 'data/ham10000',
                'd7d': 'data/derm7pt',
                }
        data_root_map_mod1 = {
                'ros': 'data/ham10000',
                'vm': 'v05/data_rescaled/ham10000',
                'd7d': 'v05/data_rescaled/derm7pt',
                }
        data_root_map_mod2 = {
                'ros': 'data/ham10000',
                'vm': 'data/ham10000',
                'd7d': 'v05/data_removedark/derm7pt',
                }
        data_root_map_mod3 = {
                'ros': 'data/ham10000',
                'vm': 'v05/data_rescaled/ham10000',
                'd7d': 'v05/data_removedark2_rescaled/derm7pt',
                }
        data_root_map_mod4 = {
                'ros': 'data/ham10000',
                'vm': 'v05/data_rescaled_adjcolor/ham10000',
                'd7d': 'v05/data_removedark2_rescaled_adjcolor/derm7pt',
                }
        dsets = data_root_map.keys()
        outroot = f'v05/m{p.model_n}/m{p.model_n}_{p.version}_seed{p.seed}'
        # treat _modX
        if p.version.endswith('_mod1'):
            p.version = p.version[:-5]  # outroot still contains modX
            data_root_map = data_root_map_mod1
        elif p.version.endswith('_mod2'):
            p.version = p.version[:-5]  # outroot still contains modX
            data_root_map = data_root_map_mod2
        elif p.version.endswith('_mod3'):
            p.version = p.version[:-5]  # outroot still contains modX
            data_root_map = data_root_map_mod3
        elif p.version.endswith('_mod4'):
            p.version = p.version[:-5]  # outroot still contains modX
            data_root_map = data_root_map_mod4
        # train
        if p.stage == 'train':
            for d in dsets:
                if p.version == 'cv':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --use-aug 0 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 0 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_sam1':
                    # default sam params, no amp, half batch-size
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size // 2} --weighted --use-aug 1 --sam --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_sam2':
                    # tuned sam: lr=5e-5, wd=0, sam_rho=0.03, amp, full batch-sz
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --lr 5e-5 --sam --sam-rho 0.03 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_sam3':
                    # default sam params, amp, full batch-size
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --sam --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_sam4':
                    # tuned sam: lr=5e-5, wd=3e-4, sam_rho=0.05, amp
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --lr 5e-5 --weight-decay 3e-4 --sam --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_wd1e-5':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --weight-decay 1e-5 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_wd1e-4':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --weight-decay 1e-4 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_wd1e-3':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --weight-decay 1e-3 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_wd1e-5':   #
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 0 --weight-decay 1e-5 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_wd3e-5':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 0 --weight-decay 3e-5 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_wd1e-3':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 0 --weight-decay 1e-3 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_adamw':
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 0 --lr 5e-5 --weight-decay 5e-3 --seed {p.seed} {outroot}/train_{d}'
                elif p.version == 'cv_wt_aug_adamw':    # for m21
                    cc = f'v05_proc.py train --model-n {p.model_n} --dsd {dsd_map[d]} --data-root {data_root_map[d]} --batch-size {p.batch_size} --weighted --use-aug 1 --lr 5e-5 --weight-decay 2e-3 --seed {p.seed} {outroot}/train_{d}'
                else:
                    raise NotImplementedError('version')
                self.execute(cc)
        # predict
        if p.stage == 'pred':
            for dt in dsets:    # train
                for dp in dsets:    # predict
                    if dt == dp:
                        cc = f'v05_proc.py predict --model-n {p.model_n} --model-dir {outroot}/train_{dt} --dsd {dsd_map[dp]} --data-root {data_root_map[dp]} --oof --batch-size {p.batch_size} --version final {outroot}/train_{dt}/{dp}_oof'
                    else:
                        cc = f'v05_proc.py predict --model-n {p.model_n} --model-dir {outroot}/train_{dt} --dsd {dsd_map[dp]} --data-root {data_root_map[dp]} --batch-size {p.batch_size} --version final {outroot}/train_{dt}/{dp}'
                    self.execute(cc)
        # calibrate-predict-proba
        if p.stage == 'fit-proba':
            for d in dsets:
                cc = f'v05_proc.py calibrate-predict-proba --beta 0.3 --n-bins 30 --data {outroot}/train_{d}/{d}_oof/softmax_m{p.model_n}.csv --out-model-path {outroot}/train_{d}/model_proba_m{p.model_n}.h5 {outroot}/train_{d}/{d}_oof/proba_m{p.model_n}.csv'
                self.execute(cc)
        # predict-proba
        if p.stage == 'pred-proba':
            for dt in dsets:    # train
                for dp in dsets:    # predict
                    if dt == dp:
                        continue
                    cc = f'v05_proc.py predict-proba --data {outroot}/train_{dt}/{dp}/softmax_m{p.model_n}.csv --model-path {outroot}/train_{dt}/model_proba_m{p.model_n}.h5 {outroot}/train_{dt}/{dp}/proba_m{p.model_n}.csv'
                    self.execute(cc)


class GenerateMask(scr.SubCommand):
    name = 'generate-mask'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root [default: %(default)s]')
        parser.add_argument('--segmenter-model-dir',
                default='segmenter',
                help='segmenter model directory')
        parser.add_argument('out_root',
                help='output root directory for maps')

    def run(self):
        sys.path.insert(0, str(self_dir / '../src'))
        from segmenter import Segmenter     # need tensorflow, eg conda env v03
        #
        p = self.p
        data_root = Path(p.data_root)
        out_root = Path(p.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        seg_params = Segmenter.load_segmentation_model(
                model_dir=p.segmenter_model_dir)
        seg = Segmenter(*seg_params)
        
        dsd = read_csv(p.dsd)
        dsd.sort_values(by=['name'], inplace=True)

        paths = dsd['path']
        
        for img_pathname in tqdm(paths):
            img = imageio.imread(data_root / img_pathname)
            mask = seg.run([img])[0]
            if 0:
                if mask.sum() == 0: # empty mask: replace by a circle
                    rr, cc = skimage.draw.circle(
                                mask.shape[0]//2,
                                mask.shape[1]//2,
                                int(min(mask.shape)*0.4),
                                )
                    mask[rr, cc] = True

            out_path = (out_root / img_pathname).with_suffix('.png')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_img = mask.astype(np.uint8) * 255
            imageio.imwrite(out_path, out_img)

class Recrop(scr.SubCommand):
    name = 'recrop'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root')
        parser.add_argument('--segmenter-model-dir',
                default='segmenter',
                help='segmenter model directory')
        parser.add_argument('--mask-root',
                required=True,
                help='mask root')
        parser.add_argument('--ref-dsd',
                required=True,
                help='reference dataset descriptor csv file path')
        parser.add_argument('--ref-mask-root',
                required=True,
                help='reference mask root')
        parser.add_argument('out_root',
                help='output data root directory')

    def run(self):
        sys.path.insert(0, str(self_dir / '../src'))
        from segmenter import Segmenter     # need tensorflow, eg conda env v03
        #
        bins = 200
        p = self.p
        data_root = Path(p.data_root)
        out_root = Path(p.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        seg_params = Segmenter.load_segmentation_model(
                model_dir=p.segmenter_model_dir)
        seg = Segmenter(*seg_params)
        
        dsd = read_csv(p.dsd)
        dsd.sort_values(by=['name'], inplace=True)

        cdf_ref = self.get_cdf(p.ref_dsd, p.ref_mask_root, bins)
        cdf_dset = self.get_cdf(p.dsd, p.mask_root, bins) 

        paths = dsd['path']
        for img_pathname in tqdm(paths):
            img = imageio.imread(data_root / img_pathname)
            mask = seg.run([img])[0]
            # fraction
            x = mask.sum() / (mask.shape[0] * mask.shape[1])
            cdf_dset_x = cdf_dset[1][np.nonzero(cdf_dset[0] > x)[0][0]] / bins
            cdf_dset_x = np.clip(cdf_dset_x, 0, 1)
            idx_cdf_ref = np.nonzero(cdf_ref[1] > cdf_dset_x*bins)[0][0]
            xx = cdf_ref[0][idx_cdf_ref]
            lin_factor = sqrt(xx / x)
            if lin_factor > 1:
                props = skimage.measure.regionprops(mask.astype(int))[0]
                r0, c0, r1, c1 = props.bbox
                Lr, Lc = mask.shape
                alpha_r = (Lr / lin_factor - Lr) / (r1 - r0 - Lr)
                alpha_c = (Lc / lin_factor - Lc) / (c1 - c0 - Lc)
                alpha_r = np.clip(alpha_r, 0, 1)
                alpha_c = np.clip(alpha_c, 0, 1)
                rr0 = int(round(alpha_r * r0))
                rr1 = int(round(Lr - alpha_r * (Lr - r1)))
                cc0 = int(round(alpha_c * c0))
                cc1 = int(round(Lc - alpha_c * (Lc - c1)))
                assert 0 <= rr0 <= r0 <= r1 <= rr1 <= Lr
                assert 0 <= cc0 <= c0 <= c1 <= cc1 <= Lc
                if 0:
                    mask2 = mask[rr0:rr1, cc0:cc1]
                    L2r, L2c = mask2.shape
                    props2 = skimage.measure.regionprops(mask2.astype(int))[0]
                    #print(f'frac={props.area / (Lr*Lc):.3f}  frac2={props2.area / (L2r*L2c):.3f}')
                    print(f'ratio: {(props2.area / (L2r*L2c)) / (props.area / (Lr*Lc)):.3f} target={xx / x:.3f}  --  {r0} {r1} {Lr} -- {c0} {c1} {Lc}')
                out_img = img[rr0:rr1, cc0:cc1]
            else:
                # shrink image, pad appropriately
                props = skimage.measure.regionprops(mask.astype(int))[0]
                r0, c0, r1, c1 = props.bbox
                Lr, Lc = mask.shape
                if r1 - r0 - Lr == 0:
                    dr_half = int(round(0.5 * (1/lin_factor - 1) * Lr))
                    rr0 = -dr_half
                    rr1 = Lr + dr_half
                else:
                    alpha_r = (Lr / lin_factor - Lr) / (r1 - r0 - Lr)
                    rr0 = int(round(alpha_r * r0))
                    rr1 = int(round(Lr - alpha_r * (Lr - r1)))
                if c1 - c0 - Lc == 0:
                    dc_half = int(round(0.5 * (1/lin_factor - 1) * Lc))
                    cc0 = -dc_half
                    cc1 = Lc + dc_half
                else:
                    alpha_c = (Lc / lin_factor - Lc) / (c1 - c0 - Lc)
                    cc0 = int(round(alpha_c * c0))
                    cc1 = int(round(Lc - alpha_c * (Lc - c1)))
                #print('old:', r0, r1, Lr, '--', c0, c1, Lc)
                #print('new:', rr0, rr1, '--', cc0, cc1)
                assert rr0 <= 0 <= r0 <= r1 <= Lr <= rr1
                assert cc0 <= 0 <= c0 <= c1 <= Lc <= cc1
                # padded with edge values
                img2 = skimage.util.pad(img,
                        ((-rr0, rr1-Lr), (-cc0, cc1-Lc), (0,0)), mode='edge')
                # blur (for padded area)
                img3 = skimage.filters.gaussian(img2, sigma=Lr/100,
                        multichannel=True, preserve_range=True).astype(np.uint8)
                # restore original area unblurred
                img3[-rr0:-rr0+Lr, -cc0:-cc0+Lc] = img
                out_img = img3
            #out_path = (out_root / img_pathname).with_suffix('.png')
            print(f'{img_pathname}  {lin_factor} -- {rr0} {rr1} {cc0} {cc1}  -- {Lr} {Lc}')
            out_path = out_root / img_pathname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_path, out_img, quality=95)

    @classmethod
    def get_cdf(cls, dsd_pathname, data_rootname, bins):
        import v05_fig
        dat = v05_fig.AreaHistogram_calc(dsd_pathname, data_rootname)
        histo, edg = np.histogram(dat, range=[0,1], bins=bins, density=True)
        val = np.cumsum(histo)
        xx = [edg[i_//2] for i_ in range(len(edg)*2)]
        yy = [0] + [val[i_//2] for i_ in range(len(val)*2)] + [bins+1]
        return xx, yy


class RemoveDarkBoundary(scr.SubCommand):
    name = 'remove-dark-boundary'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root')
        parser.add_argument('--segmenter-model-dir',
                default='segmenter',
                help='segmenter model directory')
        parser.add_argument('--extra',
                type=int, default=0,
                help='number of extra pixels')
        parser.add_argument('out_root',
                help='output data root directory')

    def run(self):
        sys.path.insert(0, str(self_dir / '../src'))
        from segmenter import Segmenter     # need tensorflow, eg conda env v03
        #
        p = self.p
        bins = 200
        data_root = Path(p.data_root)
        out_root = Path(p.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        seg_params = Segmenter.load_segmentation_model(
                model_dir=p.segmenter_model_dir)
        seg = Segmenter(*seg_params)
        
        dsd = read_csv(p.dsd)
        #dsd.sort_values(by=['name'], inplace=True)

        paths = dsd['path']
        for img_pathname in tqdm(paths):
            print(img_pathname)
            img = imageio.imread(data_root / img_pathname)
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            mask = seg.run([img])[0]
            # select non-lesion area except near borders
            select = np.ones(img.shape[:2], dtype=bool)
            select[:select.shape[0]//10] = 0
            select[select.shape[0] - select.shape[0]//10:] = 0
            select[:,:select.shape[1]//10] = 0
            select[:,select.shape[1] - select.shape[1]//10:] = 0
            select2 = select.copy()
            select = select * (1 - mask)
            if select.sum() == 0:
                select = select2
            # 0.8 * (0.02 quantile of val (in hsv) in selected area)
            val_select = np.where(select, hsv[:,:,2], 0)
            idx = np.nonzero(val_select)
            thres = 0.8 * np.quantile(val_select[idx[0], idx[1]], 0.02)
            # corner coords
            r, c = np.nonzero(hsv[:,:,2] > thres)
            # top left
            i = (5*r + c).argmin()
            r_tl, _ = r[i], c[i]
            i = (r + 5*c).argmin()
            _, c_tl = r[i], c[i]
            # top right
            i = (5*r - c).argmin()
            r_tr, _ = r[i], c[i]
            i = (r - 5*c).argmin()
            _, c_tr = r[i], c[i]
            # bottom left
            i = (-5*r + c).argmin()
            r_bl, _ = r[i], c[i]
            i = (-r + 5*c).argmin()
            _, c_bl = r[i], c[i]
            # bottom right
            i = (-5*r - c).argmin()
            r_br, _ = r[i], c[i]
            i = (-r - 5*c).argmin()
            _, c_br = r[i], c[i]
            # crop corners
            r0 = max(r_tl, r_tr)
            r1 = min(r_bl, r_br)
            c0 = max(c_tl, c_bl)
            c1 = min(c_tr, c_br)
            if 0:
                # do not crop mask
                props = skimage.measure.regionprops(mask.astype(int))[0]
                mr0, mc0, mr1, mc1 = props.bbox
                r0 = min(r0, mr0)
                r1 = max(r1, mr1-1)
                c0 = min(c0, mc0)
                c1 = max(c1, mc1-1)
            Lr, Lc, _ = img.shape
            if p.extra:
                # crop 2 more pixels, except when no cropping
                if r0 > 0:
                    r0 += p.extra
                if r1 < Lr - 1:
                    r1 -= p.extra
                if c0 > 0:
                    c0 += p.extra
                if c1 < Lc - 1:
                    c1 -= p.extra
            #
            assert 0 <= r0 < r1 < Lr
            assert 0 <= c0 < c1 < Lc
            # output: crop
            out_img = img[r0 : r1+1, c0 : c1+1]
            out_path = out_root / img_pathname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_path, out_img, quality=95)
            # test img: line is just outside crop
            test_img = img.copy()
            mark = np.array([0, 192, 192])
            if r0 > 0:
                test_img[r0-1, :] = mark
            if r1 < Lr - 1:
                test_img[r1+1, :] = mark
            if c0 > 0:
                test_img[:, c0-1] = mark
            if c1 < Lc - 1:
                test_img[:, c1+1] = mark
            out_path = out_root / 'remove_dark_boundary_test' / img_pathname.replace('/', '_')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_path, test_img, quality=95)

class AdjustColor(scr.SubCommand):
    name = 'adjust-color'

    @classmethod
    def add_arguments(self, parser):
        parser.add_argument('--dsd',
                required=True,
                help='dataset descriptor csv file path')
        parser.add_argument('--data-root',
                required=True,
                help='data root')
        parser.add_argument('out_root',
                help='output data root directory')

    def run(self):
        p = self.p
        data_root = Path(p.data_root)
        out_root = Path(p.out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        dsd = read_csv(p.dsd)
        paths = dsd['path']
        for img_pathname in tqdm(paths):
            print(img_pathname)
            img = imageio.imread(data_root / img_pathname)
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            # adjust
            if 'd7d' in p.dsd:
                hue = hsv[:,:,0]
                hue -= 14
                hue[hue > 179] -= (255 - 179)
                hsv[:,:,0] = hue
                #
                sat = hsv[:,:,1]
                sat = np.interp(sat, [0, 15, 100, 255], [0, 30, 100, 255])
                sat = np.round(sat).astype(np.uint8)
                hsv[:,:,1] = sat
                #
                val = hsv[:,:,2]
                calib = np.array([
                        [0, 0],
                        [60, 80],
                        [80, 100],
                        [100, 120],
                        [120, 130],
                        [140, 143],
                        [160, 153],
                        [180, 163],
                        [200, 173],
                        [220, 189],
                        [230, 198.3],
                        [240, 211],
                        [250, 224],
                        [255, 255],
                        ])
                val = np.interp(val, calib[:,0], calib[:,1])
                val = np.round(val).astype(np.uint8)
                hsv[:,:,2] = val

            if 'vm' in p.dsd:
                hue = hsv[:,:,0]
                hue += 8
                hue[hue > 179] -= 180
                hsv[:,:,0] = hue
                #
                sat = hsv[:,:,1]
                sat = np.interp(sat, [0, 12, 160, 255], [0, 16, 164, 255])
                sat = np.round(sat).astype(np.uint8)
                hsv[:,:,1] = sat
                #
                val = hsv[:,:,2]
                calib = np.array([
                        [0, 0],
                        [140, 140],
                        [175, 180],
                        [195, 200],
                        [218, 220],
                        [255, 255],
                        ])
                val = np.interp(val, calib[:,0], calib[:,1])
                val = np.round(val).astype(np.uint8)
                hsv[:,:,2] = val

            out_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            out_path = out_root / img_pathname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_path, out_img, quality=95)

main = scr.main_multi(__name__)

if __name__ == '__main__':
    main()

# vim: set sw=4 sts=4 expandtab :
