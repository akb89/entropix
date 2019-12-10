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
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        vocab1 = tuple1[2]
        name2 = tuple2[0]
        model2 = tuple2[1]
        vocab2 = tuple2[2]
        com_xp.dump_ndim_rmse(name1, name2, model1, model2, vocab1, vocab2,
                              NDIM, OUTPUT_DIRPATH, SCALE)
