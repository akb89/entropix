"""Compute average mean and median of seq-sampled dimensions across corpora."""
import os
import multiprocessing
import functools
import itertools
from collections import defaultdict

import numpy as np

from tqdm import tqdm

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
        reduce=True, limit=5, rewind=False, kfolding=False, kfold_size=0,
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
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc', 'enwiki']
    #MODEL_NAMES = ['enwiki07', 'oanc']
    DATASETS = ['men', 'simlex']
    #NUM_THREADS = 1
    NUM_THREADS = 35
    NUM_ITER = 10
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process, SVD_DIRPATH)
        for name, dataset, dims in tqdm(pool.imap_unordered(
            process, items(MODEL_NAMES, DATASETS, NUM_ITER)), total=NUM_ITER*len(MODEL_NAMES)*len(DATASETS)):
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
            print('{} & {}\\pm{} & {}\\pm{} & {}\\pm{} & {}\\pm{} & {}\\pm{} & {}\\pm{}'.format(
                key,
                int(round(np.mean(results[key]['men']['median']))),
                int(round(np.std(results[key]['men']['median']))),
                int(round(np.mean(results[key]['men']['mean']))),
                int(round(np.std(results[key]['men']['mean']))),
                int(round(np.mean(results[key]['men']['ninety']))),
                int(round(np.std(results[key]['men']['ninety']))),
                int(round(np.mean(results[key]['simlex']['median']))),
                int(round(np.std(results[key]['simlex']['median']))),
                int(round(np.mean(results[key]['simlex']['mean']))),
                int(round(np.std(results[key]['simlex']['mean']))),
                int(round(np.mean(results[key]['simlex']['ninety']))),
                int(round(np.std(results[key]['simlex']['ninety'])))), file=rs_stream)
