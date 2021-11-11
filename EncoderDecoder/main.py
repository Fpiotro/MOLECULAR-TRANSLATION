# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import Levenshtein
from sklearn.model_selection import train_test_split
from Trainer_LSTMATT import *
from preprocess_LSTMATT import *
import models_LSTMATT
from utils_LSTMATT import *

import warnings
warnings.filterwarnings('ignore')

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
config.read(INI_PATH.joinpath("main.ini"))
train_path = config['Main_parameters']['train_path']
tokenizer_path = config['Main_parameters']['tokenizer_path']
debug = (config['Main_parameters']['debug'] == 'True')
random_seed = int(config['Main_parameters']['random_seed'])
batch_size = int(config['Main_parameters']['batch_size'])

# ====================================================
# Main
# ====================================================
if __name__ == '__main__':

    train = pd.read_pickle(train_path)
    tokenizer = torch.load(tokenizer_path)

    # Random seed
    seed_torch(seed=random_seed)

    # Mode Debug
    if debug:
        train = train.sample(n=5000, random_state=random_seed).reset_index(drop=True)
        train , val = train_test_split(train, random_state=random_seed)

    else:
        train, val = train_test_split(train, test_size=0.10, random_state=random_seed)
        train.reset_index(drop=True)
        val.reset_index(drop=True)

    # Dataset
    train_dataset = TrainDataset(train,  tokenizer, transform=get_transforms(data='train'))
    val_dataset = TrainDataset(val,  tokenizer, transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=bms_collate)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            collate_fn=bms_collate)

    config = ConfigParser()
    config.read(INI_PATH.joinpath("params.ini"))

    # general parameters
    params = {'data_name': config['Data_parameters']['data_name'],
            'emb_dim': int(config['Model_parameters']['emb_dim']),
            'attention_dim': int(config['Model_parameters']['attention_dim']),
            'decoder_dim': int(config['Model_parameters']['decoder_dim']),
            'dropout': float(config['Model_parameters']['dropout']),
            'start_epoch': int(config['Training_parameters']['start_epoch']),
            'epochs': int(config['Training_parameters']['epochs']),
            'epochs_since_improvement': int(config['Training_parameters']['epochs_since_improvement']),
            'encoder_lr': float(config['Training_parameters']['encoder_lr']),
            'decoder_lr': float(config['Training_parameters']['decoder_lr']),
            'grad_clip': (config['Training_parameters']['grad_clip']=='True'),
            'best_cross': float(config['Training_parameters']['best_cross']),
            'print_freq': int(config['Training_parameters']['print_freq']),
            'fine_tune_encoder': (config['Training_parameters']['fine_tune_encoder']=='True'),
            'trained_weights': (config['Training_parameters']['trained_weights']=='True'),
            'trained_weights_path': config['Training_parameters']['trained_weights_path'],
            'checkpoint': (config['Training_parameters']['checkpoint']=='True'),
            'checkpoint_path':config['Training_parameters']['checkpoint_path']
            }

    la = Trainer(params)
    la.train_val_model(train_loader, val_loader)