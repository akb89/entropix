"""Evaluate a given Numpy distributional model against the MEN dataset."""
import os
import logging

import math
import scipy as sp
import numpy as np

__all__ = ('evaluate_distributional_space')

logger = logging.getLogger(__name__)

MEN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'MEN_dataset_natural_form_full')


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return sp.stats.spearmanr(x, y)[0]


def _cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


def _get_men_pairs_and_sim():
    pairs = []
    humans = []
    with open(MEN_FILEPATH, 'r', encoding='utf-8') as men_stream:
        for line in men_stream:
            line = line.rstrip('\n')
            items = line.split()
            pairs.append((items[0], items[1]))
            humans.append(float(items[2]))
    return pairs, humans


def _load_vocabulary(vocab_filepath):
    vocab = {}
    with open(vocab_filepath, 'r', encodig='utf-8') as vocab_stream:
        for line in vocab_stream:
            line = line.strip()
            vocab[line.split('\t')[1]] = line.split('\t')[0]
    return vocab


def evaluate_distributional_space(model_filepath, vocab_filepath):
    """Evaluate a numpy model against the MEN dataset."""
    logger.info('Checking embeddings quality against MEN similarity ratings')
    logger.info('Loading vocabulary...')
    vocab = _load_vocabulary(vocab_filepath)
    logger.info('Loading distributional space from {}'.format(model_filepath))
    model = np.load(model_filepath)
    pairs, humans = _get_men_pairs_and_sim()
    system_actual = []
    human_actual = []
    count = 0
    for (first, second), human in zip(pairs, humans):
        if first not in vocab or second not in vocab:
            logger.error('Could not find one of more pair item in model '
                         'vocabulary: {}, {}'.format(first, second))
            continue
        sim = _cosine_similarity(model[vocab[first]], model[vocab[second]])
        system_actual.append(sim)
        human_actual.append(human)
        count += 1
    spr = _spearman(human_actual, system_actual)
    logger.info('SPEARMAN: {} calculated over {} items'.format(spr, count))
