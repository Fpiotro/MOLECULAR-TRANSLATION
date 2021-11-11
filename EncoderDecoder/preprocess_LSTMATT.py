# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd

import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import albumentations as A
from albumentations import (Compose, OneOf, Normalize, Resize)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from configparser import ConfigParser
import sys

from utils_LSTMATT import *
import pathlib

# ====================================================
# Path
# ====================================================

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
INI_PATH = BASE_PATH.joinpath("ini").resolve()

# ====================================================
# preprocessing.ini
# ====================================================
config = ConfigParser()
config.read(INI_PATH.joinpath("preprocessing.ini"))
size = int(config['Preprocessing_parameters']['size_image'])

# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.labels = df['InChI_text'].values
        self.transform_lstm = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = (image/255).astype(np.float32)

        #Transformation before Resnet
        augmented_ = self.transform_lstm(image=image)
        image = augmented_['image']

        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label_length = len(label)
        label_length = torch.LongTensor([label_length])
        return image, torch.LongTensor(label), label_length


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = (image/255).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

def bms_collate(batch):
    imgs, labels, label_lengths = [], [], []
    for data_point in batch:
        imgs.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.stoi["<pad>"])
    return torch.stack(imgs), labels, torch.stack(label_lengths).reshape(-1, 1)

def get_transforms(*, data):

    if data == 'train':
        return Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2(),])

    elif data == 'valid':
        return Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],), ToTensorV2(),])
