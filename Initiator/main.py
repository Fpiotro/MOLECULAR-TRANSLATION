# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd
import sys
sys.path.append('F://bms-molecular-translation')

import Levenshtein
from sklearn.model_selection import train_test_split

from preprocess_InChI import *
import models_InChI
from Trainer_InChI import *
from utils_InChI import *

import warnings
warnings.filterwarnings('ignore')

# ====================================================
# Main
# ====================================================
if __name__ == '__main__':

    # Train Data
    train = pd.read_pickle(r"F://bms-molecular-translation//train_natom.pkl")
    debug=False

    # Random seed
    seed_torch(seed=42)

    # Mode Debug
    if debug:
        train = train.sample(n=1000, random_state=42).reset_index(drop=True)
        train , val = train_test_split(train, random_state=42)

    else:
        train, val = train_test_split(train, random_state=42)
        train.reset_index(drop=True)
        val.reset_index(drop=True)

    # Dataset
    train_dataset = TrainDataset(train,transform=get_transforms(data='train'))
    val_dataset = ValDataset(val,transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=True,
                            pin_memory=True)

    config = ConfigParser()
    config.read('F://bms-molecular-translation//ini//params.ini')

    # general parameters
    params = {'data_name': config['Data_parameters']['data_name'],
            'encoded_image_size': int(config['Model_parameters']['encoded_image_size']),
            'dropout': float(config['Model_parameters']['dropout']),
            'output_dim': int(config['Model_parameters']['output_dim']),
            'start_epoch': int(config['Training_parameters']['start_epoch']),
            'epochs': int(config['Training_parameters']['epochs']),
            'epochs_since_improvement': int(config['Training_parameters']['epochs_since_improvement']),
            'model_lr': float(config['Training_parameters']['model_lr']),
            'grad_clip': (config['Training_parameters']['grad_clip']=='True'),
            'best_mse': float(config['Training_parameters']['best_mse']),
            'print_freq': int(config['Training_parameters']['print_freq']),
            'fine_tune': (config['Training_parameters']['fine_tune']=='True'),
            'checkpoint': (config['Training_parameters']['checkpoint']=='True'),
            'checkpoint_path':config['Training_parameters']['checkpoint_path']
            }

    # Train
    mol = Trainer(params)
    mol.train_val_model(train_loader, val_loader)

