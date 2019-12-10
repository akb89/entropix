"""SVD-TOP-30."""

import common as com_xp


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp201.results'
    START = 0
    END = 30
    SCALE = 1e4  # scaling factor for RMSE
    print('Running entropix XP#201 on SVD-TOP-30')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    com_xp.launch_xp(MODEL_NAMES, SVD_DIRPATH, START, END, SCALE,
                     RESULTS_FILEPATH)
