# ====================================================
# Library
# ====================================================
from configparser import ConfigParser

config = ConfigParser()
config['Preprocessing_parameters'] = {}
config['Preprocessing_parameters']['size_image'] = '224'

with open('F://bms-molecular-translation//ini//preprocessing.ini', 'w') as configfile:
    config.write(configfile)

config = ConfigParser(allow_no_value=True)
config['Data_parameters'] = {}
config['Data_parameters']['data_name'] = 'test_09042021'

config['Model_parameters'] = {}
config['Model_parameters']['encoded_image_size'] = '2'
config['Model_parameters']['dropout'] = '0.1'
config['Model_parameters']['output_dim'] = '12'

config['Training_parameters'] = {}
config['Training_parameters']['start_epoch'] = '0'
config['Training_parameters']['epochs'] = '3'
config['Training_parameters']['epochs_since_improvement'] = '0'
config['Training_parameters']['model_lr'] = '1e-4'
config['Training_parameters']['grad_clip'] = 'False'
config['Training_parameters']['best_mse'] = '1'
config['Training_parameters']['print_freq'] = '1'
config['Training_parameters']['fine_tune'] = 'False'
config['Training_parameters']['checkpoint'] = 'True'
config['Training_parameters']['checkpoint_path'] = 'F://bms-molecular-translation//checkpoint_train//checkpoint_test_09042021.pth.tar'

with open('F://bms-molecular-translation//ini//params.ini', 'w') as configfile:
    config.write(configfile)