"""Evaluate a given Numpy distributional model against the MEN dataset."""
import os
import logging

import numpy as np
from scipy import stats
import scipy.spatial as spatial

import entropix.utils.data as dutils

__all__ = ('evaluate_distributional_space', 'load_words_and_sim_')

logger = logging.getLogger(__name__)

MEN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'MEN_dataset_natural_form_full')

SIMLEX_EN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'resources', 'SimLex-999.txt')

SIMVERB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'resources', 'SimVerb-3500.txt')


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return stats.spearmanr(x, y)[0]


def get_men_pairs_and_sim():
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


def get_simlex_pairs_and_sim():
    left = []
    right = []
    sim = []
    with open(SIMLEX_EN_FILEPATH, 'r', encoding='utf-8') as simlex_stream:
        for line in simlex_stream:
            line = line.rstrip('\n')
            items = line.split()
            left.append(items[0])
            right.append(items[1])
            sim.append(float(items[3]))
    return left, right, sim


def get_simverb_pairs_and_sim():
    left = []
    right = []
    sim = []
    with open(SIMVERB_FILEPATH, 'r', encoding='utf-8') as simverb_stream:
        for line in simverb_stream:
            line = line.rstrip('\n')
            items = line.split()
            left.append(items[0])
            right.append(items[1])
            sim.append(float(items[3]))
    return left, right, sim


def load_words_and_sim_(vocab_filepath, dataset):
    if dataset not in ['men', 'simlex', 'simverb']:
        raise Exception('Unsupported dataset {}'.format(dataset))
    logger.info('Loading vocabulary...')
    idx_to_word = dutils.load_vocab_mapping(vocab_filepath)
    word_to_idx = {v: k for k, v in idx_to_word.items()}
    if dataset == 'men':
        left, right, sim = get_men_pairs_and_sim()
    elif dataset == 'simlex':
        left, right, sim = get_simlex_pairs_and_sim()
    elif dataset == 'simverb':
        left, right, sim = get_simverb_pairs_and_sim()
    else:
        raise Exception('Unsupported dataset {}'.format(dataset))
    left_idx = []
    right_idx = []
    f_sim = []
    for l, r, s in zip(left, right, sim):
        if l in word_to_idx and r in word_to_idx:
            left_idx.append(word_to_idx[l])
            right_idx.append(word_to_idx[r])
            f_sim.append(s)
    return left_idx, right_idx, f_sim
    # left_idx = [word_to_idx[word] for word in left]
    # right_idx = [word_to_idx[word] for word in right]
    # return left_idx, right_idx, sim


def evaluate(model, left_idx, right_idx, sim):
    left_vectors = model[left_idx]
    right_vectors = model[right_idx]
    cos = 1 - spatial.distance.cdist(left_vectors, right_vectors, 'cosine')
    diag = np.diagonal(cos)
    spr = _spearman(sim, diag)
    return spr


def evaluate_distributional_space(model_filepath, vocab_filepath, dataset):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Checking embeddings quality against {} similarity ratings'
                .format(dataset))
    left_idx, right_idx, sim = load_words_and_sim_(vocab_filepath, dataset)
    logger.info('Loading distributional space from {}'.format(model_filepath))
    model = np.load(model_filepath)
    men_spr = evaluate(model, left_idx, right_idx, sim)
    # logger.info('Cosine distribution stats on MEN:')
    # logger.info('   Min = {}'.format(diag.min()))
    # logger.info('   Max = {}'.format(diag.max()))
    # logger.info('   Average = {}'.format(diag.mean()))
    # logger.info('   Median = {}'.format(np.median(diag)))
    # logger.info('   STD = {}'.format(np.std(diag)))
    logger.info('SPEARMAN: {}'.format(men_spr))
