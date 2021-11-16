# ====================================================
# Library
# ====================================================
import os
import re
import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm
import pathlib
tqdm.pandas()

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
input_path = config['Data_parameters']['data_path_input']
output_path = config['Data_parameters']['data_path_output']

if not os.path.isdir(output_path + "data"):
    os.mkdir(output_path + "data")

# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv(r"{}/train_labels.csv".format(input_path)).copy()

# ====================================================
# Preprocess functions
# ====================================================
def get_train_file_path(image_id):
    return "{}/train/{}/{}/{}/{}.png".format(input_path, image_id[0], image_id[1], image_id[2], image_id)

# ====================================================
# main
# ====================================================
def main():
    # ====================================================
    # preprocess train.csv
    # ====================================================
    df['file_path'] = df['image_id'].progress_apply(get_train_file_path)
    df.to_pickle("{}/data/train_labels_p.pkl".format(output_path))
    sys.stderr.write('Data saved')

if __name__ == '__main__':
    main()
