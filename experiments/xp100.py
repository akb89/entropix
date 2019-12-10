"""Compare RMSE every N set of dims on the SVD of two distinct corpora."""

import itertools

import common as com_xp


if __name__ == '__main__':
    NDIM = 30
    OUTPUT_DIRPATH = '/home/kabbach/entropix/models/frontiers/data/'
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    START = 0
    END = 10000
    SCALE = 1e4
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for name1, model1, vocab1, name2, model2, vocab2 in itertools.combination(models, 2):
        com_xp.dump_ndim_rmse(name1, name2, model1, model2, vocab1, vocab2,
                              NDIM, OUTPUT_DIRPATH, SCALE)
