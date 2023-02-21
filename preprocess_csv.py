import pandas as pd
import numpy as np
import os

def mk_dir(path):
    try:
        os.mkdir(path)
    except:
        print(f'{path} exists!')

if __name__ == '__main__':
    root = 'h3\\sdqfd_normal'
    new_root = 'new_1l2b2r\\new_G-SDQFD_normal_15'
    mk_dir(new_root)
    file = pd.read_csv(f'{root}\h3.csv').reset_index(drop=True).drop(columns=['Step'])
    for col in file:
        if 'MIN' in col or 'MAX' in col:
            continue
        npy_file = []
        for val in file[col]:
            npy_file.append([val])
        npy_file.append([0])
        print(len(npy_file),npy_file)
        mk_dir(f'{new_root}//{col[:2]}')
        mk_dir(f'{new_root}//{col[:2]}\\info')
        with open(f'{new_root}//{col[:2]}\\info\\eval_rewards.npy', 'wb') as f:
            np.save(f,npy_file)