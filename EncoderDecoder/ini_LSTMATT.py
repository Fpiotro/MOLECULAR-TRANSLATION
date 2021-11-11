# ====================================================
# Library
# ====================================================
from configparser import ConfigParser
import pathlib

# ====================================================
# Path
# ====================================================

# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
INI_PATH = BASE_PATH.joinpath("ini").resolve()

# ====================================================
# Ini
# ====================================================
config = ConfigParser()
config['Preprocessing_parameters'] = {}
config['Preprocessing_parameters']['size_image'] = '224'

with open(INI_PATH.joinpath("preprocessing.ini"), 'w') as configfile:
    config.write(configfile)

config = ConfigParser(allow_no_value=True)
config['Data_parameters'] = {}
config['Data_parameters']['data_name'] = 'LSTMATT_18052021'

config['Model_parameters'] = {}
config['Model_parameters']['emb_dim'] = '256'
config['Model_parameters']['attention_dim'] = '256'
config['Model_parameters']['decoder_dim'] = '512'
config['Model_parameters']['dropout'] = '0.5'

config['Training_parameters'] = {}
config['Training_parameters']['start_epoch'] = '0'
config['Training_parameters']['epochs'] = '1'
config['Training_parameters']['epochs_since_improvement'] = '0'
config['Training_parameters']['encoder_lr'] = '1e-4'
config['Training_parameters']['decoder_lr'] = '2e-4'
config['Training_parameters']['grad_clip'] = 'False'
config['Training_parameters']['best_cross'] = '1000'
config['Training_parameters']['print_freq'] = '1'
config['Training_parameters']['fine_tune_encoder'] = 'True'
config['Training_parameters']['trained_weights'] = 'False'
config['Training_parameters']['trained_weights_path'] = '../input/weights/trained_res.pth'
config['Training_parameters']['checkpoint'] = 'True'
config['Training_parameters']['checkpoint_path'] = '../input/modlstmatt/checkpoint_LSTMATT_18052021.pth (2).tar'

with open(INI_PATH.joinpath("params.ini"), 'w') as configfile:
    config.write(configfile)

config = ConfigParser()
config['Main_parameters'] = {}
config['Main_parameters']['train_path'] = '../input/prep-train/train2.pkl'
config['Main_parameters']['tokenizer_path'] = '../input/tokenizer/tokenizer2.pth'
config['Main_parameters']['debug'] = 'False'
config['Main_parameters']['random_seed'] = '42'
config['Main_parameters']['batch_size'] = '32'

with open(INI_PATH.joinpath("main.ini"), 'w') as configfile:
    config.write(configfile)