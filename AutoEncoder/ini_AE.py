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
config['Data_parameters']['data_name'] = 'AE_2021'
config['Data_parameters']['data_path_input'] = ''
config['Data_parameters']['data_path_output'] = ''

config['Preprocessing_parameters'] = {}
config['Preprocessing_parameters']['size_image'] = '224'

config['Model_parameters'] = {}
config['Model_parameters']['n_channels'] = '1'
config['Model_parameters']['output_dim'] = '3'

config['Training_parameters'] = {}
config['Training_parameters']['start_epoch'] = '0'
config['Training_parameters']['epochs'] = '10'
config['Training_parameters']['epochs_since_improvement'] = '0'
config['Training_parameters']['model_lr'] = '5e-5'
config['Training_parameters']['grad_clip'] = 'False'
config['Training_parameters']['best_mse'] = '1000'
config['Training_parameters']['print_freq'] = '1'
config['Training_parameters']['checkpoint'] = 'False'
config['Training_parameters']['checkpoint_path'] = './checkpoint/checkpoint_AE_2021.pth.tar'

config['Main_parameters'] = {}
config['Main_parameters']['debug'] = 'False'
config['Main_parameters']['random_seed'] = '42'
config['Main_parameters']['batch_size'] = '32'

with open(INI_PATH.joinpath("params.ini"), 'w') as configfile:
    config.write(configfile)
