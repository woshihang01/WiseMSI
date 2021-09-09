from __future__ import print_function, division
import os

import openslide
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F
import random, tqdm, staintools
from PIL import Image
import h5py

from random import randrange


def get_stain_normalizer(path='stainColorNormalization/template.png', method='macenko'):
    target = staintools.read_image(path)
    target = staintools.LuminosityStandardizer.standardize(target)
    normalizer = staintools.StainNormalizer(method=method)
    normalizer.fit(target)
    return normalizer


normalizer = get_stain_normalizer()


def apply_stain_norm(tile, normalizer):
    to_transform = np.array(tile).astype('uint8')
    to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
    transformed = normalizer.transform(to_transform)
    transformed = Image.fromarray(transformed)
    return transformed


def patches_gen_transforms(training=False):
    if training:
        trnsfrms_val = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
    else:
        trnsfrms_val = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
    return trnsfrms_val


def eval_transforms(pretrained=False):
    if pretrained:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

    trnsfrms_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
    )

    return trnsfrms_val


class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 file_path,
                 pretrained=False,
                 custom_transforms=None,
                 target_patch_size=-1,
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.pretrained = pretrained
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, target_patch_size)
        else:
            self.target_patch_size = None

        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['imgs']
        for name, value in dset.attrs.items():
            print(name, value)

        print('pretrained:', self.pretrained)
        print('transformations:', self.roi_transforms)
        if self.target_patch_size is not None:
            print('target_size: ', self.target_patch_size)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
                 file_path,
                 wsi,
                 pretrained=False,
                 normalization=False,
                 custom_transforms=None,
                 custom_downsample=1,
                 target_patch_size=-1
                 ):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            pretrained (bool): Use ImageNet transforms
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
            target_patch_size (int): Custom defined image size before embedding
        """
        self.pretrained = pretrained
        self.wsi = wsi
        self.normalization = normalization
        if not custom_transforms:
            self.roi_transforms = eval_transforms(pretrained=pretrained)
        else:
            self.roi_transforms = custom_transforms

        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)
            if target_patch_size > 0:
                self.target_patch_size = (target_patch_size,) * 2
            elif custom_downsample > 1:
                self.target_patch_size = (self.patch_size // custom_downsample,) * 2
            else:
                self.target_patch_size = None
        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('target patch size: ', self.target_patch_size)
        print('pretrained: ', self.pretrained)
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img).unsqueeze(0)
        return img, coord


class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]


class Whole_Slide_Patches_Gen(Dataset):
    def __init__(self, csv_file_path,
                 h5_file_path,
                 training,
                 label_dicts=[{}, {}, {}],
                 patch_level=0,
                 patch_size=512,
                 ):
        slide_data = pd.read_csv(csv_file_path)
        self.label_dicts = label_dicts
        self.num_classes = [len(set(label_dict.values())) for label_dict in self.label_dicts]
        self.patch_level = patch_level
        self.patch_size = patch_size
        slide_data = self.df_prep(slide_data, self.label_dicts, ['label'])
        slide_data_dict = slide_data[['case_id', 'label']].set_index('case_id')['label'].to_dict()
        self.slides = slide_data['slide_id'].tolist()
        self.cases = slide_data['case_id'].tolist()
        self.paths = slide_data['path'].tolist()
        self.h5_file_path = h5_file_path
        self.training = training
        self.roi_transforms = patches_gen_transforms()
        self.wsi_list = []
        self.data_list = []
        for slide_id, case_id, path in tqdm.tqdm(zip(self.slides, self.cases, self.paths)):
            with h5py.File(os.path.join(self.h5_file_path, '.'.join(case_id.split('.')[:-1]) + '.h5'), "r") as f:
                dset = f['coords'][()].tolist()
                for coord in dset:
                    self.wsi_list.append(case_id)
                    self.data_list.append(
                        [case_id, coord[0], coord[1], slide_data_dict[case_id], slide_id, path + '/' + case_id])
        self.length = len(self.data_list)
        # self.summary()

    @staticmethod
    def df_prep(data, label_dicts, label_cols):
        if label_cols[0] != 'label':
            data['label'] = data[label_cols[0]].copy()

        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dicts[0][key]

        for idx, (label_dict, label_col) in enumerate(zip(label_dicts[1:], label_cols[1:])):
            print(label_dict, label_col)
            data[label_col] = data[label_col].map(label_dict)

        return data

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            with openslide.OpenSlide(self.data_list[idx][5]) as slide:
                img = slide.read_region((self.data_list[idx][1], self.data_list[idx][2]), self.patch_level,
                                        (self.patch_size, self.patch_size)).convert('RGB')
                img = apply_stain_norm(img, normalizer)
                img = self.roi_transforms(img)
        except Exception as e:
            print(self.data_list[idx][5])
        return img, self.data_list[idx][3]

    def return_splits(self, csv_path, max_nums):
        assert max_nums
        all_splits = pd.read_csv(csv_path)
        train_patient = all_splits['train'].dropna().tolist()
        val_patient = all_splits['val'].dropna().tolist()
        test_patient = all_splits['test'].dropna().tolist()
        dataframe = DataFrame(self.data_list)
        train_df = dataframe[dataframe[4].isin(train_patient)]
        val_df = dataframe[dataframe[4].isin(val_patient)]
        test_df = dataframe[dataframe[4].isin(test_patient)]
        train_split = Whole_Slide_Patches_Split(train_df, max_nums[0], self.training, self.patch_level,
                                                self.patch_size)
        val_split = Whole_Slide_Patches_Split(val_df, max_nums[1], self.training, self.patch_level,
                                              self.patch_size)
        test_split = Whole_Slide_Patches_Split(test_df, max_nums[2], self.training, self.patch_level,
                                               self.patch_size)

        return train_split, val_split, test_split


class Whole_Slide_Patches_Split(Whole_Slide_Patches_Gen):
    def __init__(self, df,
                 max_num,
                 training,
                 patch_level=0,
                 patch_size=512,
                 custom_transforms=None,
                 ):
        assert type(df) == DataFrame
        if max_num != float('INF'):
            df = df.groupby(4).sample(n=max_num, random_state=1, replace=True)
        self.data_list = df.values.tolist()
        self.wsi_list = df[0].values.tolist()
        self.length = len(self.data_list)
        self.num_classes = df[3].nunique()
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.training = training
        if not custom_transforms:
            self.roi_transforms = patches_gen_transforms()
        else:
            self.roi_transforms = custom_transforms

    def __len__(self):
        return self.length


if __name__ == '__main__':
    dataset = Whole_Slide_Patches_Gen(r'C:\Code\WiseMSI\dataset_csv\dataset_train_test2_test3_msi_95tumor.csv',
                                      r'C:\RESULTS_TUMOR_STAIN_NORM_95\patches',
                                      training=True,
                                      label_dicts=[{'MSS': 0, 'MSI-H': 1}],
                                      )
    train_dataset, val_dataset, test_dataset = dataset.return_splits(
        r'C:\Code\WiseMSI\splits\msi_classifier_100\splits_0.csv', [100, 100, float('INF')])
    dl = DataLoader(train_dataset, batch_size=33, shuffle=True)
    for x, y in dl:
        print('emmm')
