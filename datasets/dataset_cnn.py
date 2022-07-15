import random

from torch.utils.data import Dataset
import pandas as pd
import os
import PIL.Image as Image
import torch
import numpy as np
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class WSI_Patches_Image(Dataset):
    def __init__(self, data_list, patch_image_path, training=True, patch_size=256, maximum=float('inf')):
        self.data_list = []
        self.labels = []
        self.wsi_list = []
        if training:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        trans = [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]
        if patch_size != 256:
            trans.insert(0, transforms.Resize(256))
        self.transform = transforms.Compose(
            trans
        )
        for data in data_list:
            if 'normal' not in os.listdir(patch_image_path + '/' + data):
                print(patch_image_path + '/' + data + ' is error')
                raise NotImplementedError
            elif 'tumor' not in os.listdir(patch_image_path + '/' + data):
                print(patch_image_path + '/' + data + ' is error')
                raise NotImplementedError
            elif len(os.listdir(patch_image_path + '/' + data)) != 2:
                print(patch_image_path + '/' + data + ' is error')
                raise NotImplementedError
            image_list = os.listdir(patch_image_path + '/' + data + '/normal')
            if len(image_list) > maximum:
                random.shuffle(image_list)
                image_list = image_list[:maximum]
            for image in image_list:
                self.data_list.append(patch_image_path + '/' + data + '/normal/' + image)
                self.labels.append(0)
                self.wsi_list.append(data)
            image_list = os.listdir(patch_image_path + '/' + data + '/tumor')
            if len(image_list) > maximum:
                random.shuffle(image_list)
                image_list = image_list[:maximum]
            for image in image_list:
                self.data_list.append(patch_image_path + '/' + data + '/tumor/' + image)
                self.labels.append(1)
                self.wsi_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = Image.open(self.data_list[idx])
        if self.transform:
            data = self.transform(data)
        return data, self.labels[idx]
