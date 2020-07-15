import itertools

import numpy as np
from tqdm import tqdm

import common as com_xp
import entropix.core.aligner as aligner

if __name__ == '__main__':
    START = 0
    END = 10000
    # SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    MODEL_NAMES = ['enwiki07', 'oanc']
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        vocab1 = tuple1[2]
        name2 = tuple2[0]
        model2 = tuple2[1]
        vocab2 = tuple2[2]
        print('Processing models {} and {}'.format(name1, name2))
        z, t, _ = aligner.align_vocab(model1, model2, vocab1, vocab2)
        with open('{}-{}-overlap.dat'.format(name1, name2), 'w', encoding='utf-8') as outs:
            for col1, col2 in tqdm(zip(z.T, t.T), total=z.shape[1]):
                col1 = np.abs(col1)
                col2 = np.abs(col2)
                diff = np.abs(col1 - col2)
                overlap = np.sum(diff)
                print(overlap, file=outs)
