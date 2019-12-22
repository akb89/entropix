"""Compute average mean and median of seq-sampled dimensions across corpora."""
import os
import multiprocessing
import functools
import itertools

from collections import defaultdict

import common as com_xp
import entropix.utils.metrix as metrix
from entropix.core.sampler import Sampler


def _process(dirpath, name, dataset):
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
    return name, dataset, dims


if __name__ == '__main__':
    #SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    #MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    MODEL_NAMES = ['enwiki07', 'oanc']
    DATASETS = ['men', 'simlex']
    NUM_THREADS = 1
    results = defaultdict(defaultdict(list))
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process)
        for name, dataset, dims in pool.imap_unordered(
            process, itertools.product(MODEL_NAMES, DATASETS)):
            pass
