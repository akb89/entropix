"""MEN-RAND-30."""

import common as com_xp


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp204.results'
    DIMS_DIRPATH = '/home/kabbach/entropix/models/frontiers/limit/'
    START = 0
    END = 10000
    SCALE = 1e4  # scaling factor for RMSE
    print('Running entropix XP#203 on MEN-LIMIT-30')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    com_xp.launch_xp(MODEL_NAMES, SVD_DIRPATH, START, END, SCALE,
                     RESULTS_FILEPATH, randomize=True,
                     dims_dirpath=DIMS_DIRPATH, dataset='men')
