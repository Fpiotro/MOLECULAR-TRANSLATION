# ====================================================
# Library
# ====================================================
import os
import re
import numpy as np
import pandas as pd
import sys
from utils_InChI import save_obj
from tqdm.auto import tqdm
tqdm.pandas()


# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv(r"F://bms-molecular-translation//train_labels.csv").copy()

atom = {'B': 0, 'Br': 1, 'C': 2, 'Cl': 3, 'F': 4, 'H': 5, 'I': 6, 'N': 7, 'O': 8, 'P': 9, 'S': 10, 'Si': 11}

# Save to decode sequence from model
save_obj(atom, 'atom')

# ====================================================
# Preprocess functions
# ====================================================
def split_InChI(s):
    res  = re.findall('[A-Z][a-z]?|[0-9]+', s)
    l = ['0']*(len(atom))
    for i,j in enumerate(res):
        try:
            index = atom[j]
            try:
                nb = int(res[i+1])
                l[index]= str(nb)
            except:
                l[index]='1'
        except:
            pass
    return "/".join(l)

def get_train_file_path(image_id):
    return "F:/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)

def f(x):
    return np.array(x.split('/')).astype(float)

# ====================================================
# main
# ====================================================
def main():
    # ====================================================
    # preprocess train.csv
    # ====================================================
    df['Ft'] = df['InChI'].progress_apply(lambda x: x.split('/')[1])
    df['NbAt'] = df['Ft'].progress_apply(split_InChI)
    df['file_path'] = df['image_id'].progress_apply(get_train_file_path)
    df.to_pickle("F://bms-molecular-translation//train_labels_p.pkl")
    sys.stderr.write('Data saved')

    m = np.array(list(map(f,df.NbAt.values)))
    data  = pd.DataFrame(m,columns=atom.keys())
    data['file_path'] = df.file_path
    data.to_pickle("F://bms-molecular-translation//train_natom.pkl")
    sys.stderr.write('Data saved')

if __name__ == '__main__':
    main()

