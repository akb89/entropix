"""Evaluation metrics utilities."""

import logging
import math
import numpy as np
from scipy import stats
import scipy.spatial as spatial

logger = logging.getLogger(__name__)

__all__ = ('get_combined_spr_rmse', 'get_spr_correlation', 'get_rmse',
           'get_both_spr_rmse', 'purity')


def purity(y_true, y_pred):
    """
    Calculate purity for given true and predicted cluster labels.
    Parameters
    ----------
    y_true: array, shape: (n_samples, 1)
      True cluster labels
    y_pred: array, shape: (n_samples, 1)
      Cluster assingment.
    Returns
    -------
    purity: float
      Calculated purity.
    """
    assert len(y_true) == len(y_pred)
    true_clusters = np.zeros(shape=(len(set(y_true)), len(y_true)))
    pred_clusters = np.zeros_like(true_clusters)
    for id, cl in enumerate(set(y_true)):
        true_clusters[id] = (y_true == cl).astype("int")
    for id, cl in enumerate(set(y_pred)):
        pred_clusters[id] = (y_pred == cl).astype("int")
    M = pred_clusters.dot(true_clusters.T)
    return 1. / len(y_true) * np.sum(np.max(M, axis=1))


# Note: this is scipy's spearman, without tie adjustment
def spearman(x, y):
    return stats.spearmanr(x, y)[0]


def root_mean_square_error(x, y):
    """Return root mean squared error"""
    return np.sqrt(((x - y) ** 2).mean())


def cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


def get_gensim_model_sim(model, left_words, right_words):
    sim = []
    for left, right in zip(left_words, right_words):
        if left not in model.wv.vocab or right not in model.wv.vocab:
            logger.error('Could not find one of more pair item in model '
                         'vocabulary: {}, {}'.format(left, right))
            continue
        sim.append(cosine_similarity(model.wv[left], model.wv[right]))
    return sim


def similarity(left_vectors, right_vectors, distance):
    """Compute euclidian or cosine similarity between two matrices."""
    if distance not in ['cosine', 'euclidean']:
        raise Exception('Unsupported distance: {}'.format(distance))
    if left_vectors.shape != right_vectors.shape:
        raise Exception(
            'Cannot compute similarity from numpy arrays of different shape: '
            '{} != {}'.format(left_vectors.shape(), right_vectors.shape()))
    if distance == 'cosine':
        dotprod = np.einsum('ij,ij->i', left_vectors, right_vectors)
        norms = np.linalg.norm(left_vectors, axis=1) * np.linalg.norm(right_vectors, axis=1)
        return dotprod / norms
    # TODO: refactor below as above
    euc = spatial.distance.cdist(left_vectors, right_vectors, 'euclidean')
    diag = np.diag(euc)
    sim = (diag - np.min(diag)) / (np.max(diag) - np.min(diag))
    return sim


def get_numpy_model_sim(model, left_idx, right_idx, dataset, distance):
    if dataset in ['men', 'simlex', 'simverb', 'ws353']:
        left_vectors = model[left_idx]
        right_vectors = model[right_idx]
    elif dataset == 'sts2012':
        left_vectors = []
        right_vectors = []
        for idx_list in left_idx:
            vec = np.sum([model[el] for el in idx_list], axis=0)
            left_vectors.append(vec)
        for idx_list in right_idx:
            vec = np.sum([model[el] for el in idx_list], axis=0)
            right_vectors.append(vec)
    return similarity(left_vectors, right_vectors, distance)


def get_rmse(model, left_idx, right_idx, sim, dataset, distance):
    if dataset not in ['men', 'simlex', 'simverb', 'ws353']:
        raise Exception('Unsupported dataset: {}'.format(dataset))
    model_sim = get_numpy_model_sim(model, left_idx, right_idx, dataset,
                                    distance)
    return root_mean_square_error(sim, model_sim)


def get_spr_correlation(model, left_idx, right_idx, sim, dataset, distance):
    model_sim = get_numpy_model_sim(model, left_idx, right_idx, dataset,
                                    distance)
    return spearman(sim, model_sim)


def get_combined_spr_rmse(model, left_idx, right_idx, sim, dataset, alpha,
                          distance):
    if alpha < 0 or alpha > 1:
        raise Exception('Invalid alpha value = {}. Should be in [0, 1]'
                        .format(alpha))
    spr = get_spr_correlation(model, left_idx, right_idx, sim, dataset,
                              distance)
    rmse = get_rmse(model, left_idx, right_idx, sim, dataset, distance)
    return alpha * spr - (1 - alpha) * rmse


def get_both_spr_rmse(model, left_idx, right_idx, sim, dataset, distance):
    return '{}#{}'.format(get_spr_correlation(model, left_idx, right_idx, sim,
                                              dataset, distance),
                          get_rmse(model, left_idx, right_idx, sim, dataset,
                                   distance))
