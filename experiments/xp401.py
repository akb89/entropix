"""Compute average mean and median of seq-sampled dimensions across corpora."""
import os
import multiprocessing
import functools
import itertools
from collections import defaultdict

import numpy as np

import common as com_xp
from entropix.core.sampler import Sampler


def items(names, datasets, num_iter):
    count = sum(1 for x in itertools.product(names, datasets)) * num_iter
    return itertools.islice(itertools.cycle(itertools.product(names, datasets)), count)


def _process(dirpath, name_and_dataset):
    name = name_and_dataset[0]
    dataset = name_and_dataset[1]
    model_filepath = os.path.join(dirpath, '{}-aligned.npy'.format(name))
    vocab_filepath = os.path.join(dirpath, '{}-aligned.vocab'.format(name))
    sampler = Sampler(
        singvectors_filepath=model_filepath, model_type='numpy',
        vocab_filepath=vocab_filepath, datasets=[dataset],
        output_basepath=os.path.join(dirpath, '{}-{}-seq'.format(name, dataset)),
        num_iter=2, shuffle=True, mode='seq', rate=None, start=0, end=10000,
        reduce=True, limit=None, rewind=False, kfolding=False, kfold_size=1,
        max_num_threads=1, debug=False, metric='both', alpha=None,
        logs_dirpath=None, distance='cosine', singvalues_filepath=None,
        sing_alpha=0, dump=False)
    dims = sampler.sample_dimensions()
    # dims = [1, 2, 10]
    return name, dataset, dims


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    #SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/models/frontiers/results/xp401.results'
    #RESULTS_FILEPATH = '/Users/akb/Github/entropix/models/frontiers/results/xp401.results'
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    #MODEL_NAMES = ['enwiki07', 'oanc']
    DATASETS = ['men', 'simlex']
    NUM_THREADS = 55
    NUM_ITER = 10
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process, SVD_DIRPATH)
        for name, dataset, dims in pool.imap_unordered(
            process, items(MODEL_NAMES, DATASETS, NUM_ITER)):
            mean = np.mean(dims)
            median = np.median(dims)
            results[name][dataset]['mean'].append(mean)
            results[name][dataset]['median'].append(median)
            print(name, dataset)
    with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as rs_stream:
        print('MODEL & MEN-MEAN & MEN-MEDIAN & SIMLEX-MEAN & SIMLEX-MEDIAN',
              file=rs_stream)
        for key in results.keys():
            print('{} & {}\\pm{} & {}\\pm{} & {}\\pm{} & {}\\pm{}'.format(
                key,
                np.mean(results[key]['men']['mean']),
                np.std(results[key]['men']['mean']),
                np.mean(results[key]['men']['median']),
                np.std(results[key]['men']['median']),
                np.mean(results[key]['simlex']['mean']),
                np.std(results[key]['simlex']['mean']),
                np.mean(results[key]['simlex']['median']),
                np.std(results[key]['simlex']['median'])), file=rs_stream)
