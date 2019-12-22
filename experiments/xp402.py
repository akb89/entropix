"""Compute mean and median of limit-sampled dims across corpora."""
import os
import itertools
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import common as com_utils

if __name__ == '__main__':
    #DIMS_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    DIMS_DIRPATH = '/Users/akb/Gitlab/frontiers/data/'
    #RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp402.results'
    RESULTS_FILEPATH = '/Users/akb/Github/entropix/models/frontiers/results/xp402.results'
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc', 'enwiki']
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    # MODEL_NAMES = ['enwiki07', 'oanc']
    DATASETS = ['men', 'simlex']
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, dataset in tqdm(itertools.product(MODEL_NAMES, DATASETS),
                              total=len(MODEL_NAMES)*len(DATASETS)):
        dims_filepath = os.path.join(DIMS_DIRPATH, '{}-{}.dims'.format(name, dataset))
        dims = com_utils.load_dims(dims_filepath)
        mean = np.mean(dims)
        median = np.median(dims)
        ninety = np.percentile(dims, 90)
        results[name][dataset]['mean'].append(mean)
        results[name][dataset]['median'].append(median)
        results[name][dataset]['ninety'].append(ninety)
    with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as rs_stream:
        print('MODEL & MEN-MEDIAN & MEN-MEAN & MEN-90p & SIMLEX-MEDIAN & SIMLEX-MEAN & SIMLEX-90p',
              file=rs_stream)
        for key in results.keys():
            print('{} & {} & {} & {} & {} & {} & {}'.format(
                key,
                int(round(np.mean(results[key]['men']['median']))),
                int(round(np.mean(results[key]['men']['mean']))),
                int(round(np.mean(results[key]['men']['ninety']))),
                int(round(np.mean(results[key]['simlex']['median']))),
                int(round(np.mean(results[key]['simlex']['mean']))),
                int(round(np.mean(results[key]['simlex']['ninety'])))), file=rs_stream)
