# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import (Compose, OneOf, Normalize, Resize)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from configparser import ConfigParser

from utils_AE import *
import pathlib

# ====================================================
# Path
# ====================================================
BASE_PATH = pathlib.Path(__file__).parent.resolve()
INI_PATH = BASE_PATH.joinpath("ini").resolve()

# ====================================================
# params.ini
# ====================================================
config = ConfigParser()
config.read(INI_PATH.joinpath("params.ini"))
size = int(config['Preprocessing_parameters']['size_image'])

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df['InChI'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        labels = self.labels[idx]
        image = cv2.imread(file_path)
        
        label = draw_mol(labels, image.shape[1], image.shape[0]).astype(np.float32)
        
        drop_bonds = np.random.randint(2)
        
        if drop_bonds:
            img = 255 - random_molecule_image(labels, image.shape[1], image.shape[0])
        else:
            img = label
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img[img>0] = 255

        # random boolean mask for which values will be changed
        mask = np.random.choice(2,size = img.shape, p=[0.50,0.50]).astype(np.bool)
        mask_b = np.random.choice(2,size = img.shape, p=[0.9995,0.0005]).astype(np.bool)
        
        # random matrix the same shape of your data
        r = np.zeros(img.shape)
        b = np.ones(img.shape)*255
        
        # use your mask to replace values in your input array
        img[mask] = r[mask]
        img[mask_b] = b[mask_b]
        img = (img/255).astype(np.float32)
        
        augmented = self.transform(image=label)
        label = augmented['image']
        augmented_ = self.transform(image=img)
        img = augmented_['image']
        return img, label

class ValDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df['InChI'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        file_path = self.file_paths[idx]
        labels = self.labels[idx]
        image = cv2.imread(file_path)
        
        label = draw_mol(labels, image.shape[1], image.shape[0]).astype(np.float32)

        if drop_bonds:
            img = 255 - random_molecule_image(labels, image.shape[1], image.shape[0])
        else:
            img = label
            
        img = cv2.cvtColor(255 - img, cv2.COLOR_RGB2GRAY)

        img[img>0] = 255

        # random boolean mask for which values will be changed
        mask = np.random.choice(2,size = img.shape, p=[0.4,0.6]).astype(np.bool)
        mask_b = np.random.choice(2,size = img.shape, p=[0.9995,0.0005]).astype(np.bool)
        
        # random matrix the same shape of your data
        r = np.zeros(img.shape)
        b = np.ones(img.shape)*255
        
        # use your mask to replace values in your input array
        img[mask] = r[mask]
        img[mask_b] = b[mask_b]
        img = (img/255).astype(np.float32)
        
        augmented = self.transform(image=label)
        label = augmented['image']
        augmented_ = self.transform(image=img)
        img = augmented_['image']
        return img, label


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = ((255-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))/255).astype(np.float32)
        h, w = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        augmented = self.transform(image=image)
        image = augmented['image']
        return image
    
def get_transforms(*, data):

    if data == 'train':
        return Compose([
            Resize(size, size),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(size, size),
            ToTensorV2(),
        ])
