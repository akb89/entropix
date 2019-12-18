"""SVD-RAND-30."""

import common as com_xp


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp202.results'
    START = 0
    END = 30
    SCALE = 1e4  # scaling factor for RMSE
    NRUNS = 10
    print('Running entropix XP#202 on SVD-RAND-30')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    com_xp.launch_xp(MODEL_NAMES, SVD_DIRPATH, START, END, SCALE,
                     RESULTS_FILEPATH, randomize=True, nruns=NRUNS)
