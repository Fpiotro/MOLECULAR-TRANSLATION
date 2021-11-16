# ====================================================
# Library
# ====================================================
from configparser import ConfigParser
import pathlib
import os

# ====================================================
# Path
# ====================================================
BASE_PATH = pathlib.Path(__file__).parent.resolve()
INI_PATH = BASE_PATH.joinpath("ini").resolve()

if not os.path.isdir(INI_PATH):
    os.mkdir(INI_PATH)

# ====================================================
# Ini
# ===================================================
config = ConfigParser(allow_no_value=True)

config['Data_parameters'] = {}
config['Data_parameters']['data_path_input'] = ''
config['Data_parameters']['data_path_output'] = ''

config['Preprocessing_parameters'] = {}
config['Preprocessing_parameters']['size_image'] = '224'

config['Model_parameters'] = {}
config['Model_parameters']['n_channels'] = '1'
config['Model_parameters']['output_dim'] = '3'

config['Training_parameters'] = {}

# Epochs
config['Training_parameters']['start_epoch'] = '0'
config['Training_parameters']['epochs'] = '10'
config['Training_parameters']['epochs_since_improvement'] = '0'

# Learning Rate
config['Training_parameters']['model_lr'] = '1e-4'
config['Training_parameters']['grad_clip'] = 'False'
config['Training_parameters']['grad_clip_value'] = ''
config['Training_parameters']['step_size_scheduler'] = '2'
config['Training_parameters']['gamma_scheduler'] = '0.1'

# Loss
config['Training_parameters']['best_mse'] = '100'

# Display
config['Training_parameters']['print_freq'] = '10'

# Checkpoint
config['Training_parameters']['checkpoint_name'] = 'AE_2021'
config['Training_parameters']['checkpoint'] = 'False'
config['Training_parameters']['checkpoint_path'] = ''

config['Main_parameters'] = {}

# Configuration
config['Main_parameters']['debug'] = 'False'
config['Main_parameters']['random_seed'] = '42'

# DataLoader
config['Main_parameters']['batch_size'] = '32'
config['Main_parameters']['shuffle'] = 'True'
config['Main_parameters']['pin_memory'] = 'True'
config['Main_parameters']['drop_last'] = 'False'

with open(INI_PATH.joinpath("params.ini"), 'w') as configfile:
    config.write(configfile)
