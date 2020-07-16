"""Compute RMSE and avg PEARSON correlations for bins."""
import itertools

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
from tqdm import tqdm

import common as com_xp
import entropix.core.aligner as aligner
import entropix.utils.metrix as metrix

if __name__ == '__main__':
    START = 0
    END = 10000
    SCALE = 1e4
    BIN_SIZE = 30
    # SVD_DIRPATH = '/home/kabbach/entropix/models/frontiers/aligned/'
    SVD_DIRPATH = '/Users/akb/Github/entropix/models/frontiers/aligned/'
    RESULTS_FILEPATH = '/Users/akb/Github/entropix/models/frontiers/results/smoothed-pearson-rmse.dat'
    # MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    MODEL_NAMES = ['enwiki07', 'oanc']
    models = com_xp.load_aligned_models(MODEL_NAMES, SVD_DIRPATH, START, END)
    for tuple1, tuple2 in itertools.combinations(models, 2):
        name1 = tuple1[0]
        model1 = tuple1[1]
        vocab1 = tuple1[2]
        name2 = tuple2[0]
        model2 = tuple2[1]
        vocab2 = tuple2[2]
        print('Processing models {} and {}'.format(name1, name2))
        z, t, _ = aligner.align_vocab(model1, model2, vocab1, vocab2)
        rmses = []
        xcorrx = []
        for idx in tqdm(range(z.shape[1])):
            if idx % BIN_SIZE == 0:
                if idx + BIN_SIZE > z.shape[1]:
                    break
                m1 = z[:, idx:idx+BIN_SIZE]
                m2 = t[:, idx:idx+BIN_SIZE]
                rmses.append(com_xp.get_rmse(m1, m2) * SCALE)
            xcorr = metrix.pearson_correlation(z[:, idx], t[:, idx])
            xcorrx.append(xcorr)
        xcorrx = np.abs(xcorrx)
        avgs = com_xp.binize(np.array(xcorrx), BIN_SIZE).mean(axis=1)
        avgs = np.log(avgs)
        rmse_hat = sig.savgol_filter(rmses, 121, 4)
        rmse_hat2 = sig.savgol_filter(rmses, 71, 3)
        rmse_hat3 = sig.savgol_filter(rmses, 71, 5)
        avg_hat = sig.savgol_filter(avgs, 141, 7)
        avg_hat2 = sig.savgol_filter(avgs, 111, 5)
        avg_hat3 = sig.savgol_filter(avgs, 211, 5)
        avg_hat4 = sig.savgol_filter(avgs, 71, 5)
        avg_hat5 = sig.savgol_filter(avgs, 71, 5)
        plt.plot(avgs, rmse_hat, color='red')
        #plt.plot(avg_hat4, rmse_hat, color='green')
        #plt.plot(avg_hat5, rmse_hat, color='brown')
        plt.plot(avg_hat, rmse_hat, color='blue')
        #plt.plot(avg_hat5, rmse_hat3, color='yellow')
        #plt.plot(avg_hat4, rmses, color='yellow')
        print(avg_hat)
        # with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as out_str:
        #
        plt.show()
        # with open('{}-{}-pearson-rmse-n{}-log.dat'
        #           .format(name1, name2, BIN_SIZE), 'w',
        #           encoding='utf-8') as outs:
        #     for avg, rmse in zip(avgs, rmses):
        #         print('{}\t{}'.format(avg, rmse), file=outs)
