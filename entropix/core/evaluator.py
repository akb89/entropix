"""Evaluate a given Numpy distributional model against the MEN dataset."""
import os
import logging

import numpy as np
from scipy import stats
import scipy.spatial as spatial

import entropix.utils.data as dutils

__all__ = ('evaluate_distributional_space')

logger = logging.getLogger(__name__)

MEN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'MEN_dataset_natural_form_full')


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return stats.spearmanr(x, y)[0]


def _get_men_pairs_and_sim():
    left = []
    right = []
    sim = []
    with open(MEN_FILEPATH, 'r', encoding='utf-8') as men_stream:
        for line in men_stream:
            line = line.rstrip('\n')
            items = line.split()
            left.append(items[0])
            right.append(items[1])
            sim.append(float(items[2]))
    return left, right, sim


def evaluate_distributional_space(model_filepath, vocab_filepath):
    """Evaluate a numpy model against the MEN dataset."""
    logger.info('Checking embeddings quality against MEN similarity ratings')
    logger.info('Loading vocabulary...')
    idx_to_word = dutils.load_vocab_mapping(vocab_filepath)
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    logger.info('Loading distributional space from {}'.format(model_filepath))
    model = np.load(model_filepath)
    left, right, sim = _get_men_pairs_and_sim()
    left_idx = [word_to_idx[word] for word in left]
    right_idx = [word_to_idx[word] for word in right]
    left_vectors = model[left_idx]
    right_vectors = model[right_idx]
    cos = 1 - spatial.distance.cdist(left_vectors, right_vectors, 'cosine')
    diag = np.diagonal(cos)
    spr = _spearman(sim, diag)
    logger.info('SPEARMAN: {} calculated over {} items'.format(spr, len(left)))
