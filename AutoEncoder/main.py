# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from Trainer_AE import *
from preprocess_AE import *
import models_AE
from utils_AE import *

import warnings
warnings.filterwarnings('ignore')

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

output_path = config['Data_parameters']['data_path_output']

debug = (config['Main_parameters']['debug'] == 'True')
random_seed = int(config['Main_parameters']['random_seed'])

batch_size = int(config['Main_parameters']['batch_size'])
shuffle = eval(config['Main_parameters']['shuffle'])
pin_memory = eval(config['Main_parameters']['pin_memory'])
drop_last = eval(config['Main_parameters']['drop_last'])

# ====================================================
# Main
# ====================================================
if __name__ == '__main__':

    # Train Data
    train = pd.read_pickle(r"{}/train_labels_p.pkl".format(output_path))

    # Random seed
    seed_torch(seed=random_seed)

    # Mode Debug
    if debug:
        train = train.sample(n=1000, random_state=random_seed).reset_index(drop=True)
        train , val = train_test_split(train, random_state=random_seed)

    else:
        train, val = train_test_split(train, test_size=0.10, random_state=random_seed)
        train.reset_index(drop=True)
        val.reset_index(drop=True)

    # Dataset
    train_dataset = TrainDataset(train, transform=get_transforms(data='train'))
    val_dataset = ValDataset(val, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              drop_last=drop_last)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            pin_memory=pin_memory,
                            drop_last=drop_last)

    # general parameters
    params = {'n_channels': int(config['Model_parameters']['n_channels']),
            'output_dim': int(config['Model_parameters']['output_dim']),
            'start_epoch': int(config['Training_parameters']['start_epoch']),
            'epochs': int(config['Training_parameters']['epochs']),
            'epochs_since_improvement': int(config['Training_parameters']['epochs_since_improvement']),
            'model_lr': float(config['Training_parameters']['model_lr']),
            'grad_clip': eval(config['Training_parameters']['grad_clip']),
            'grad_clip_value': float(config['Training_parameters']['grad_clip_value']),
            'step_size_scheduler' : int(config['Training_parameters']['step_size_scheduler']),
            'gamma_scheduler' : float(config['Training_parameters']['gamma_scheduler']),
            'best_mse': float(config['Training_parameters']['best_mse']),
            'print_freq': int(config['Training_parameters']['print_freq']),
            'checkpoint_name': config['Training_parameters']['checkpoint_name'],
            'checkpoint': (config['Training_parameters']['checkpoint']=='True'),
            'checkpoint_path': config['Training_parameters']['checkpoint_path']
            }

    mol = Trainer(params)
    mol.train_val_model(train_loader, val_loader)

    #df_test = pd.read_csv("../input/bms-molecular-translation/sample_submission.csv").copy()

    #def get_test_file_path(image_id):
      #return "../input/bms-molecular-translation/test/{}/{}/{}/{}.png".format(
          #image_id[0], image_id[1], image_id[2], image_id)

    #df_test['file_path'] = df_test['image_id'].progress_apply(get_test_file_path)

    #test_img_dataset = TestDataset(df_test,transform=get_transforms(data='valid'))

    #index = 2003

    #inp_ = (test_img_dataset[index].squeeze().numpy()*255).astype(np.uint8)
    #pred_ = mol.model(test_img_dataset[index].unsqueeze(0).to(device))
    #pred_ = (pred_.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)

    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50,100))

    ## Display the image
    #ax1.imshow(inp_)
    #ax1.set_title('Val_Input', fontsize=50)
    #ax2.imshow(pred_)
    #ax2.set_title('Prediction', fontsize=50)
