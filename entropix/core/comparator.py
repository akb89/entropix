"""Measure nearest neighbor overlap between two numpy models."""
import logging

import numpy as np
import scipy.spatial as spatial
import entropix.utils.metrix as metrix

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _get_n_nearest_neighbors(idx, model, n):
    vector = model[idx]
    sim = []
    for i in tqdm(range(model.shape[0])):
        if i == idx:
            sim.append(0)
        else:
            ivector = model[i]
            cos = metrix.cosine_similarity(vector, ivector)
            sim.append(cos)
    return set(np.argsort(sim)[::-1][:n])


def _compare_low_ram(model1, model2, n):
    variance = []
    assert model1.shape[0] == model2.shape[0]
    for idx in range(model1.shape[0]):
        neighb1 = _get_n_nearest_neighbors(idx, model1, n)
        neighb2 = _get_n_nearest_neighbors(idx, model2, n)
        variance.append(1 - len(neighb1.intersection(neighb2))/n)
    return variance


def _compare(model1, model2, n):
    try:
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
    except MemoryError:
        return _compare_low_ram(model1, model2, n)


def _align_model_vocab(model1, model2, vocab1, vocab2):
    logger.info('Aligning model vocabularies...')
    vocab_2_to_vocab1 = {idx: vocab1[word] for word, idx in vocab2.items()
                         if word in vocab1}
    _model1 = np.empty(shape=(len(vocab_2_to_vocab1), model1.shape[1]))
    idx2 = [idx for word, idx in vocab2.items() if word in vocab1]
    assert len(idx2) == len(vocab_2_to_vocab1)
    _model2 = model2[idx2, :]
    for idx, item in enumerate(sorted(idx2)):
        _model1[idx] = model1[vocab_2_to_vocab1[item]]
    return _model1, _model2


def align_vocab(model1, model2, vocab1, vocab2):
    if len(vocab1) != len(vocab2):
        return _align_model_vocab(model1, model2, vocab1, vocab2)
    for word, idx in vocab1.items():
        if word not in vocab2:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
        if vocab2[word] != idx:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
    logger.info('Processing already aligned vocabularies')
    return model1, model2


def compare(model1, model2, vocab1, vocab2, n):
    model1, model2 = align_vocab(model1, model2, vocab1, vocab2)
    variance = _compare(model1, model2, n)
    # take the average and std
    avg = np.mean(variance)
    std = np.std(variance)
    return avg, std
