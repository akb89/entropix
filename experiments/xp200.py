"""SVD-TOP-300."""

import common as com_xp


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp200.results'
    START = 0
    END = 300
    SCALE = 1e4  # scaling factor for RMSE
    print('Running entropix XP#200 on SVD-TOP-300')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc',
                   'enwiki']
    com_xp.launch_xp(MODEL_NAMES, SVD_DIRPATH, START, END, SCALE,
                     RESULTS_FILEPATH)
