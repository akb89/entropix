"""Evaluate the density (defined as ratio between zero and non-zero values) in matrices."""
import os
import itertools

import matplotlib.pyplot as plt

from tqdm import tqdm

import common as com_xp
import entropix.utils.metrix as metrix


def binize(data, bin_size):
    """Binning a numpy array."""
    return data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size)


if __name__ == '__main__':
    START = 0
    END = 10000
    SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    #SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    #MODEL_NAMES = ['enwiki07', 'oanc']
    EPSILON = 1e-4
    BIN_SIZE = 30
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        name2 = tuple2[0]
        model2 = tuple2[1]
        print('Processing models {} and {}'.format(name1, name2))
        xcorrx = []
        max_corrx = []
        offsets = []
        for col1, col2 in tqdm(zip(model1.T, model2.T), total=model1.shape[1]):
            xcorr, max_corr, offset = metrix.cross_correlation(col1, col2)
            xcorr /= metrix.xcorr_norm(col1, col2)
            max_corr /= metrix.xcorr_norm(col1, col2)
            xcorrx.append(xcorr)
            max_corrx.append(max_corr)
            offsets.append(offset)
        with open('{}-{}-xcorr.dat'.format(name1, name2), 'w', encoding='urf-8') as outs:
            for xcorr, max_corr, offset in zip(xcorrx, max_corrx, offsets):
                print('{}\t{}\t{}'.format(xcorr, max_corr, offset), file=out_str)
        # plt.plot(xcorrx)
        # plt.plot(max_corrx)
        # plt.plot(offsets)
        # plt.show()
