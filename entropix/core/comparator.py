"""Measure nearest neighbor overlap between two numpy models."""
import logging

import numpy as np
import scipy.spatial as spatial

logger = logging.getLogger(__name__)


def _compare(model1, model2, n):
    # compute cosine similarities
    logger.info('Computing cosine similarities...')
    cos1 = 1 - spatial.distance.cdist(model1, model1, 'cosine')
    cos2 = 1 - spatial.distance.cdist(model2, model2, 'cosine')
    # postprocess to avoid taking the word itself as its own nearest neighbor
    cos1 = cos1 - np.diag(np.ones(model1.shape[0]))
    cos2 = cos2 - np.diag(np.ones(model2.shape[0]))
    # compute the n-nearest neighbor indices
    logger.info('Computing nearest neighbor indices...')
    idx1 = np.argpartition(cos1, -n, axis=1)[:, -n:]
    idx2 = np.argpartition(cos2, -n, axis=1)[:, -n:]
    # sort to be able to use np.count_nonzero(A==B, axis=1)
    idx1.sort()
    idx2.sort()
    logger.info('Computing variance...')
    # compute the variance following Pierrejean and Tanguy 2018
    return 1 - np.count_nonzero(idx1 == idx2, axis=1) / n


def _align_model_vocab(model1, model2, vocab1, vocab2):
    # align model1 with model2's vocab
    vocab1_to_vocab2_idx = {idx: vocab2[word] for word, idx in vocab1.items()
                            if word in vocab2}
    _model1 = np.empty(shape=(len(vocab1_to_vocab2_idx), model1.shape[1]))
    idx2 = [idx for word, idx in vocab2.items() if word in vocab1]
    assert len(idx2) == len(vocab1_to_vocab2_idx)
    _model2 = model2[idx2, :]
    for idx, item in enumerate(sorted(idx2)):
        _model1[idx] = model1[vocab1_to_vocab2_idx[item]]
    return _model1, _model2


def align_vocab(model1, model2, vocab1, vocab2):
    if len(vocab1) != len(vocab2):
        return _align_model_vocab(model1, model2, vocab1, vocab2)
    for word, idx in vocab1.items():
        if word not in vocab2:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
        if vocab2[word] != idx:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
    return model1, model2


def compare(model1, model2, vocab1, vocab2, n):
    model1, model2 = align_vocab(model1, model2, vocab1, vocab2)
    variance = _compare(model1, model2, n)
    # take the average and std
    avg = np.mean(variance)
    std = np.std(variance)
    return avg, std
