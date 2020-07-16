"""Generate a set of aligned vocabularies for each model."""

import common as com_xp

if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/svd/'
    OUTPUT_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    START = 0
    END = 10000
    print('Aligning vocabularies across all models')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc',
                   'enwiki']
    MODELS = com_xp.load_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    com_xp.dump_aligned_models(MODELS, OUTPUT_DIRPATH)
