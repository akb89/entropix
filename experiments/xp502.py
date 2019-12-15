"""Evaluate the density (defined as ratio between zero and non-zero values) in matrices."""
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as sig

from tqdm import tqdm

import common as com_xp

def convolution(x, y):
    return np.convolve(x, y, mode='valid')[0]

def pearson_correlation(x, y):
    # x -= x.mean()
    # x /= x.std()
    # y -= y.mean()
    # y += y.std()
    return np.abs(stats.pearsonr(x, y)[0])


def cross_corr(x, y):
    n = x.size
    x -= x.mean()
    x /= x.std()
    y -= y.mean()
    y += y.std()
    # xcorr = sig.correlate(x, y, mode='full')
    xcorr = sig.correlate(x, y, mode='valid')
    return np.abs(xcorr[0])
    # dt = np.arange(1-n, n)
    # return dt[xcorr.argmax()]


def cross_correlation(x, y):
    """Return normalized cross-correlation and offset.

    Assuming x and y to be same-size arrays, in full mode, we will get a
    cross-correlation array of size x.size + y.size - 1 = 2 * n - 1
    (with n = x.size).
    To get the offset,
    """
    assert x.size == y.size
    x = (x - np.mean(x)) / (np.std(x) * len(x))
    y = (y - np.mean(y)) / (np.std(y))
    _xcorr = np.correlate(x, y, mode='full')

    return xcorr, offset, max_corr_at_offset


def binize(data, bin_size):
    """Binning a numpy array."""
    return data[:(data.size // bin_size) * bin_size].reshape(-1, bin_size)


if __name__ == '__main__':
    START = 0
    END = 10000
    #SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    #MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    MODEL_NAMES = ['enwiki07', 'oanc']
    EPSILON = 1e-4
    BIN_SIZE = 30
    # test_array = np.random.rand(1,30)[0]
    # print(test_array)
    # prod = itertools.permutations(test_array)
    # print(sum(1 for i in prod))
    # test_arr = np.array([[1, 2, 0, 3], [3, 9, 0, 4]])
    # print(np.count_nonzero(test_arr==0, axis=1))
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        name2 = tuple2[0]
        model2 = tuple2[1]
        print('Processing models {} and {}'.format(name1, name2))
        # with open('enwiki07.dim1000.txt', 'w', encoding='utf-8') as enwiki:
        #     for row in model1[:, 1000]:
        #         print(row, file=enwiki)
        # with open('oanc.dim1000.txt', 'w', encoding='utf-8') as oanc:
        #     for row in model2[:, 1000]:
        #         print(row, file=oanc)
        # plt.plot(model2[:, 1000])
        # plt.show()
        xcorrx = []
        for col1, col2 in tqdm(zip(model1.T, model2.T), total=model1.shape[1]):
            xcorr = cross_correlation(col1, col2)
            #xcorr = pearson_correlation(col1, col2)
            xcorrx.append(xcorr)
        xcorrx = np.array(xcorrx)
        # avgs = binize(xcorrx, BIN_SIZE).mean(axis=1)
        #zeros = np.count_nonzero(binize(xcorrx, BIN_SIZE) == 0, axis=1)
        # plt.plot(avgs)
        #print(xcorrx)
        #print(zeros)
        plt.plot(xcorrx[1000:])
        plt.show()
