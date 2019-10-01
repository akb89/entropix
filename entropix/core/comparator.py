"""Measure nearest neighbor overlap between two numpy models."""

import numpy as np
import scipy.spatial as spatial


def _compare(model1, model2, n):
    # compute cosine similarities
    cos1 = 1 - spatial.distance.cdist(model1, model1, 'cosine')
    cos2 = 1 - spatial.distance.cdist(model2, model2, 'cosine')
    # postprocess to avoid taking the word itself as its own nearest neighbor
    cos1 = cos1 - np.diag(np.ones(model1.shape[0]))
    cos2 = cos2 - np.diag(np.ones(model2.shape[0]))
    # compute the n-nearest neighbor indices
    idx1 = np.argpartition(cos1, -n, axis=1)[:, -n:]
    idx2 = np.argpartition(cos2, -n, axis=1)[:, -n:]
    # sort to be able to use np.count_nonzero(A==B, axis=1)
    idx1.sort()
    idx2.sort()
    # compute the variance following Pierrejean and Tanguy 2018
    return 1 - np.count_nonzero(idx1 == idx2, axis=1) / n


def align_vocab(vocab1, vocab2):
    if len(vocab1) != len(vocab2):
        raise Exception('Cannot process unidentical vocabularies')
    for word, idx in vocab1.items():
        if word not in vocab2:
            raise Exception('Cannot process unidentical vocabularies')
        if vocab2[word] != idx:
            raise Exception('Cannot process unidentical vocabularies')
    return vocab1


def compare(model1, model2, vocab1, vocab2, n):
    vocab = align_vocab(vocab1, vocab2)
    variance = _compare(model1, model2, n)
    # take the average and std
    avg = np.mean(variance)
    std = np.std(variance)
    return avg, std
