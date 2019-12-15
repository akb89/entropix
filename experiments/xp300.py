"""Dump absolute Pearson correlations between singular vectors of two models."""
import itertools

from tqdm import tqdm

import common as com_xp
import entropix.utils.metrix as metrix

if __name__ == '__main__':
    START = 0
    END = 10000
    # SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    # MODEL_NAMES = ['enwiki07', 'oanc']
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        name2 = tuple2[0]
        model2 = tuple2[1]
        print('Processing models {} and {}'.format(name1, name2))
        with open('{}-{}-pearson.dat'.format(name1, name2), 'w', encoding='utf-8') as outs:
            for col1, col2 in tqdm(zip(model1.T, model2.T), total=model1.shape[1]):
                xcorr = abs(metrix.pearson_correlation(col1, col2))
                print(xcorr, file=outs)
