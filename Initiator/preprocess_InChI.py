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
from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose, Blur
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from configparser import ConfigParser

# ====================================================
# preprocessing.ini
# ====================================================
config = ConfigParser()
config.read('F://bms-molecular-translation//ini//preprocessing.ini')
size = int(config['Preprocessing_parameters']['size_image'])

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df.drop('file_path',axis=1).values
        self.max_labels = np.max(self.labels, axis=0)
        self.min_labels = np.min(self.labels, axis=0)
        self.save_max_min()
        self.transform = transform
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = (255 - cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = self.labels[idx]
        label = torch.from_numpy((label-self.min_labels)/(self.max_labels-self.min_labels))
        return image, label

    def save_max_min(self):
        torch.save(torch.from_numpy(self.max_labels), "F://bms-molecular-translation//obj//data_train_max.pth")
        torch.save(torch.from_numpy(self.min_labels), "F://bms-molecular-translation//obj//data_train_min.pth")

class ValDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.labels = df.drop('file_path',axis=1).values
        self.max_labels, self.min_labels = self.load_max_min()
        self.transform = transform
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = (255 - cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.from_numpy(self.labels[idx]).float()
        label = ((label-self.min_labels)/(self.max_labels-self.min_labels)).float()
        return image, label

    def load_max_min(self):
        return torch.load("F://bms-molecular-translation//obj//data_train_max.pth"), torch.load("F://bms-molecular-translation//obj//data_train_min.pth")

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
        image = (255 - cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def get_transforms(*, data):

    if data == 'train':
        return Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return Compose([
            Resize(size, size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
