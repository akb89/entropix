"""Export singular values for plot."""

import os
import numpy as np
from scipy import sparse

import entropix.utils.metrix as metrix

if __name__ == '__main__':
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/ppmi/'
    DATA_DIRPATH = '/Users/akb/Gitlab/frontiers/data/'
    START = 0
    END = 300
    print('Running entropix XP#601')
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc',
    #                'enwiki']
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'enwiki4', 'bnc', 'enwiki']
    # MODEL_NAMES = ['enwiki07']
    for name in MODEL_NAMES:
        singvalues_filepath = os.path.join(SVD_DIRPATH, '{}-singvalues.npy'.format(name))
        singvalues = np.load(singvalues_filepath)
        with open(os.path.join(DATA_DIRPATH, '{}-singvalues.dat'.format(name)), 'w', encoding='utf-8') as output_str:
            for singvalue in singvalues:
                print(singvalue, file=output_str)
