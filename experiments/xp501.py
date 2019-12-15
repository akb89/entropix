"""Evaluate the density (defined as ratio between zero and non-zero values) in matrices."""
import os
import itertools

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.signal as sig

from tqdm import tqdm

import common as com_xp

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result


def cross_correlation(x, y):
    n = x.size
    x -= x.mean()
    x /= x.std()
    y -= y.mean()
    y += y.std()
    xcorr = sig.correlate(x, y)
    dt = np.arange(1-n, n)
    return np.abs(dt[xcorr.argmax()])

if __name__ == '__main__':
    START = 0
    END = 10000
    #SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    #MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    MODEL_NAMES = ['enwiki07', 'oanc']
    EPSILON = 1e-4
    BIN_SIZE = 30
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        vocab1 = tuple1[2]
        name2 = tuple2[0]
        model2 = tuple2[1]
        vocab2 = tuple2[2]
        # print(model1[0])
        # print(model1[:, 0])
        # density1 = np.nonzero(model1 > EPSILON)[0].shape[0] / (model1.shape[0] * model1.shape[1])
        # density2 = np.nonzero(model2 > EPSILON)[0].shape[0] / (model2.shape[0] * model2.shape[1])
        # print('{} density = {}'
        #       .format(name1, density1))
        # print('{} density = {}'
        #       .format(name2, density2))
        # print('Processing model {}'.format(name1))
        # for col in model1.T:
        #     col_density = np.nonzero(col > EPSILON)[0].shape[0] / model1.shape[0]
        #     print(col_density)
        print('Processing models {} and {}'.format(name1, name2))
        # A = np.abs(model1 - model2)
        # B = np.sum(A, axis=0)
        # C = np.abs(model2 - model1)
        # D = np.sum(C, axis=0)
        # # A = ((model1 - model2) + 2) / 4
        # plt.plot(B)
        # plt.plot(D)
        # plt.show()
        # print(B)
        # A = model1 > EPSILON
        # B = model2 > EPSILON
        # C = A == B
        # D = np.sum(C, axis=0)
        # # counter = 0
        # # minval = model1.shape[1]
        # # min_array = []
        # # for sumval in D:
        # #     counter += 1
        # #     if sumval < minval:
        # #         minval = sumval
        # #     if counter == BIN_SIZE:
        # #         min_array.append(minval)
        # #         minval = model1.shape[1]
        # #         counter = 0
        # plt.plot(D[:50])
        # # reduced = min_array[:4]
        # # plt.plot(reduced)
        # plt.plot(model1[:, 0])
        # y = model1[:, 1000]
        # print(y.mean())
        # plt.plot(y)
        # plt.show()
        # avgs = []
        # stds = []
        # for col in model1.T:
        #     avg = np.mean(col)
        #     avgs.append(avg)
        #     std = np.std(col)
        #     stds.append(std)
        # plt.plot(stds[1:])
        # plt.plot(avgs[1:])
        # plt.show()
        xcorrx = []
        for col1, col2 in tqdm(zip(model1.T, model2.T), total=model1.shape[1]):
            #corr = sig.correlate(col1, col2)
            xcorr = cross_correlation(col1, col2)
            xcorrx.append(xcorr)
        plt.plot(xcorrx)
        plt.show()
        # x = model1[:, 1000]
        # auto = estimated_autocorrelation(x)
        # print(auto)
        # x = model1[:, 0]
        # auto = estimated_autocorrelation(x)
        # print(auto)
        # print(model1.shape)
        # x = model1[:, 0]
        # x -= x.mean()
        # x /= x.std()
        # corr = sig.correlate(x, x)
        # print(corr)
        # print(corr.shape)
        # print(corr.mean())
        # y = model1[:, 1000]
        # y -= y.mean()
        # y /= y.std()
        # corr2 = sig.correlate(y, y)
        # print(corr2)
        # print(corr2.shape)
        # print(corr2.mean())
        # #plt.plot(corr)
        # plt.plot(corr2)
        # plt.show()
