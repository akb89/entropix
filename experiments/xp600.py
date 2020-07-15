"""Computing energy in PPMI-weighted matrices and in singular values."""

import os
import numpy as np
from scipy import sparse

import entropix.utils.metrix as metrix

if __name__ == '__main__':
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/ppmi/'
    RESULTS_FILEPATH = '/Users/akb/Github/entropix/models/frontiers/results/xp600.results'
    START = 0
    END = 300
    print('Running entropix XP#600')
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc',
    #                'enwiki']
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'enwiki4', 'bnc', 'enwiki']
    MODEL_NAMES = ['acl']
    for name in MODEL_NAMES:
        ppmi_filepath = os.path.join(SVD_DIRPATH, '{}-ppmi.npz'.format(name))
        singvalues_filepath = os.path.join(SVD_DIRPATH, '{}-singvalues.npy'.format(name))
        ppmi = sparse.load_npz(ppmi_filepath)
        singvalues = np.load(singvalues_filepath)
        total_energy = metrix.energy(ppmi)
        total_svd_energy = metrix.energy(singvalues)
        reduced_svd_energy = metrix.energy(singvalues, start=START, end=END)
        per_full = round((total_svd_energy / total_energy) * 100, 1)
        per_reduced = round((reduced_svd_energy / total_energy) * 100, 1)
        print('{}\t{}\t{}'.format(name, per_full, per_reduced))
