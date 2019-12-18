"""Compare RMSE every N set of dims on the SVD of two distinct corpora."""

import os
import numpy as np

from tqdm import tqdm

import entropix.utils.data as dutils
import entropix.utils.metrix as metrix
import entropix.core.aligner as aligner
import entropix.core.matrixor as matrixor

if __name__ == '__main__':
    SVD = ['acl.mincount-3.win-2.npy',
           'bnc.mincount-30.win-2.npy',
           'enwiki1.mincount-10.win-2.npy',
           'enwiki2.mincount-30.win-2.npy',
           'enwiki4.mincount-30.win-2.npy',
           'enwiki07.mincount-10.win-2.npy',
           'oanc.mincount-10.win-2.npy']
    NDIM = 30
    OUTPUT_DIRPATH = '/home/kabbach/entropix/models/svd/'
    loaded = []
    for x in SVD:
        x_vocab = '{}.vocab'.format(x.split('.npy')[0])
        print('Loading model {}...'.format(x))
        model, vocab = dutils.load_model_and_vocab(
            x, 'numpy', x_vocab)
        loaded.append((x, model, vocab))
    size = []
    for name1, model1, vocab1 in loaded:
        aligned_model1 = model1
        vocab = vocab1
        for name2, model2, vocab2 in loaded:
            if name1 == name2:
                continue
            assert aligned_model1.shape[1] == model2.shape[1]
            aligned_model1, _, vocab = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
        size.append(aligned_model1.shape[0])
        print('aligned vocab size = {}'.format(aligned_model1.shape[0]))
        for name2, model2, vocab2 in loaded:
            if name1 == name2:
                continue
            z, t, _ = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
            assert z.shape[0] == t.shape[0]
            results = []
            print('Computing RMSE on {} and {} for every set of {} dims...'
                  .format(name1, name2, NDIM))
            for idx in tqdm(range(z.shape[1])):
                if idx % NDIM == 0:
                    if idx + NDIM > z.shape[1]:
                        break
                    m1 = z[:, idx:idx+NDIM]
                    m2 = t[:, idx:idx+NDIM]
                    assert m1.shape[1] == m2.shape[1] == NDIM
                    T = matrixor.apply_absolute_orientation_with_scaling(m1, m2)
                    V = matrixor.apply_absolute_orientation_with_scaling(m2, m1)
                    rmse1 = metrix.root_mean_square_error(m1, T)
                    rmse2 = metrix.root_mean_square_error(m2, V)
                    avg = (rmse1 + rmse2) / 2
                    # print('rmse1 = {}, rmse2 = {}, avg = {}'.format(rmse1, rmse2, avg))
                    results.append((idx, avg))
            OUTPUT_FILEPATH = os.path.join(
                OUTPUT_DIRPATH, '{}-{}-n{}-all-aligned.dat'.format(
                    name1.split('.mincount')[0], name2.split('.mincount')[0],
                    NDIM))
            print('Saving results to {}'.format(OUTPUT_FILEPATH))
            with open(OUTPUT_FILEPATH, 'w', encoding='utf-8') as output_str:
                for idx, rmse in results:
                    print('{}\t{}'.format(idx, rmse), file=output_str)
    for x in size:
        assert x == size[0]
