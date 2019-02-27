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

STS2012_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'resources', 'STS2012.full.txt')


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
            left.append([items[0]])
            right.append([items[1]])
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
            left.append([items[0]])
            right.append([items[1]])
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
            left.append([items[0]])
            right.append([items[1]])
            sim.append(float(items[3]))
    return left, right, sim


def get_STS2012_pairs_and_sim():
    left = []
    right = []
    sim = []
    with open(STS2012_FILEPATH, 'r', encoding='utf-8') as sts_stream:
        for line in sts_stream:
            line = line.strip().split('\t')
            st1 = [x.lower() for x in line[0].split()]
            st2 = [x.lower() for x in line[1].split()]
            score = float(line[2])
            left.append(st1)
            right.append(st2)
            sim.append(score)
    return left, right, sim

def load_words_and_sim_(vocab_filepath, dataset):
    if dataset not in ['men', 'simlex', 'simverb', 'sts2012']:
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
    elif dataset == 'sts2012':
        left, right, sim = get_STS2012_pairs_and_sim()
    else:
        raise Exception('Unsupported dataset {}'.format(dataset))
    left_idx = []
    right_idx = []
    f_sim = []
    for l, r, s in zip(left, right, sim):
        l_in_word_to_idx = [x for x in l if x in word_to_idx]
        r_in_word_to_idx = [x for x in r if x in word_to_idx]
        if len(l_in_word_to_idx)/len(l) > 0.85 and\
           len(r_in_word_to_idx)/len(r) > 0.85:
            left_idx.append([word_to_idx[x] for x in l_in_word_to_idx])
            right_idx.append([word_to_idx[x] for x in r_in_word_to_idx])
            f_sim.append(s)
    return left_idx, right_idx, f_sim
    # left_idx = [word_to_idx[word] for word in left]
    # right_idx = [word_to_idx[word] for word in right]
    # return left_idx, right_idx, sim


def evaluate(model, left_idx, right_idx, sim):
    left_vectors = []
    right_vectors = []
    for idx_list in left_idx:
        vec = np.sum([model[el] for el in idx_list], axis=0)
        left_vectors.append(vec)
    for idx_list in right_idx:
        vec = np.sum([model[el] for el in idx_list], axis=0)
        right_vectors.append(vec)
    cos = 1 - spatial.distance.cdist(left_vectors, right_vectors, 'cosine')
    diag = np.diagonal(cos)
    spr = _spearman(sim, diag)
    return spr


def evaluate_distributional_space(model, vocab_filepath, dataset):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Checking embeddings quality against {} similarity ratings'
                .format(dataset))
    left_idx, right_idx, sim = load_words_and_sim_(vocab_filepath, dataset)
    men_spr = evaluate(model, left_idx, right_idx, sim)
    # logger.info('Cosine distribution stats on MEN:')
    # logger.info('   Min = {}'.format(diag.min()))
    # logger.info('   Max = {}'.format(diag.max()))
    # logger.info('   Average = {}'.format(diag.mean()))
    # logger.info('   Median = {}'.format(np.median(diag)))
    # logger.info('   STD = {}'.format(np.std(diag)))
    logger.info('SPEARMAN: {}'.format(men_spr))
