import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

from tqdm import tqdm

import imageio

class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        #image = cv2.imread(row.filepath)
        image = imageio.imread(row.filepath)
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            meta = self.csv.iloc[index][self.meta_features]
        else:
            meta = 0.
        data = (torch.tensor(image).float(), torch.tensor(meta).float())

        if self.mode == 'test':
            return data
        else:
            target = self.csv.iloc[index].target
            return data, torch.tensor(target).long()


def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

def generate_df(dsd:pd.DataFrame, use_meta:bool, data_folder, out_dim, 
        diagnosis2idx, data_root:Path, is_training=False):
    # drop fold < 0 entries
    if is_training:
        dsd = dsd[dsd['fold'] >= 0].reset_index(drop=True)
    # placeholder for all sites
    all_sites = [
            'anterior torso',
            'posterior torso',
            'lateral torso',
            'torso',
            'head/neck',
            'upper extremity',
            'lower extremity',
            'palms/soles',
            'oral/genital',
            np.nan,
            ]
    df_sites = pd.DataFrame({'anatom_site': all_sites})
    dsd_aug = pd.concat([dsd, df_sites])
    onehot = pd.get_dummies(dsd_aug['anatom_site'],
            dummy_na=True, dtype=np.uint8, prefix='site')
    df = pd.DataFrame({
        'image_name': dsd['name'],
        'filepath': str(data_root) + '/' + dsd['path'],
        })
    # sex
    try:
        df['sex'] = dsd['sex'].map({'male': 1, 'female': 0}).fillna(-1)
    except:
        df['sex'] = -1
    # age
    try:
        df['age_approx'] = (dsd['age'] / 90).fillna(0)
    except:
        df['age_approx'] = 0
    # n_images and patient_id
    if 'patient_id' in dsd.columns:
        df['patient_id'] = dsd['patient_id'].fillna(-1)
        df['n_images'] = df['patient_id'].map(df.groupby(
            ['patient_id']).image_name.count())
        df.loc[df['patient_id'] == -1, 'n_images'] = 1
    else:
        df['n_images'] = 1
    df['n_images'] = np.log1p(df['n_images'].values)

    # image_size
        # median log imagefile size of the appropriate folder. std=0.23
    df['image_size'] = {
            512: 10.989,
            768: 11.590,
            1024: 0,    # the only data_folder==1024 model does not use meta
            }[data_folder]
    df = pd.concat([df, onehot.iloc[:-len(all_sites)]], axis=1)

    if use_meta:
        meta_features = ['sex', 'age_approx', 'n_images', 'image_size',
                'site_anterior torso', 'site_head/neck', 'site_lateral torso',
                'site_lower extremity', 'site_oral/genital', 'site_palms/soles',
                'site_posterior torso', 'site_torso', 'site_upper extremity',
                'site_nan']
    else:
        meta_features = None

    if is_training and 'diagnosis' in dsd.columns:
        df['target'] = dsd['diagnosis'].map(diagnosis2idx)
        df['fold'] = dsd['fold']

    return df, meta_features


def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features

# vim: set sw=4 sts=4 expandtab :
