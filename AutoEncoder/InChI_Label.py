# ====================================================
# Library
# ====================================================
import os
import re
import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm
tqdm.pandas()

# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv(r"F://bms-molecular-translation//train_labels.csv").copy()

# ====================================================
# Preprocess functions
# ====================================================
def get_train_file_path(image_id):
    return "F:/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)

# ====================================================
# main
# ====================================================
def main():
    # ====================================================
    # preprocess train.csv
    # ====================================================
    df['file_path'] = df['image_id'].progress_apply(get_train_file_path)
    df.to_pickle("F://bms-molecular-translation//train_labels_p.pkl")
    sys.stderr.write('Data saved')

if __name__ == '__main__':
    main()
