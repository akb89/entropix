"""Evaluate a given Numpy distributional model against the MEN dataset."""
import os
import logging
import random
import math
import numpy as np
from scipy import stats
import scipy.spatial as spatial

import entropix.utils.data as dutils

__all__ = ('evaluate_distributional_space', 'load_words_and_sim',
           'load_kfold_train_test_dict')

logger = logging.getLogger(__name__)

MEN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'resources', 'MEN_dataset_natural_form_full')

SIMLEX_EN_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'resources', 'SimLex-999.txt')

SIMVERB_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'resources', 'SimVerb-3500.txt')

STS2012_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'resources', 'STS2012.full.txt')

WS353_FILEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                              'resources',
                                              'WS353.combined.txt')


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


def get_WS353_pairs_and_sim():
    left = []
    right = []
    sim = []
    with open(WS353_FILEPATH, 'r', encoding='utf-8') as ws_stream:
        for line in ws_stream:
            line = line.rstrip('\n')
            items = line.split()
            left.append(items[0])
            right.append(items[1])
            sim.append(float(items[2]))
    return left, right, sim


def load_words_and_sim(vocab_filepath, dataset, shuffle):
    if dataset not in ['men', 'simlex', 'simverb', 'sts2012', 'ws353']:
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
    elif dataset == 'ws353':
        left, right, sim = get_WS353_pairs_and_sim()
    else:
        raise Exception('Unsupported dataset {}'.format(dataset))
    left_idx = []
    right_idx = []
    f_sim = []
    if dataset in ['men', 'simlex', 'simverb', 'ws353']:
        for l, r, s in zip(left, right, sim):
            if l in word_to_idx and r in word_to_idx:
                left_idx.append(word_to_idx[l])
                right_idx.append(word_to_idx[r])
                f_sim.append(s)
    # TODO: remove hardcoded values?
    if dataset == 'sts2012':
        for l, r, s in zip(left, right, sim):
            l_in_word_to_idx = [x for x in l if x in word_to_idx]
            r_in_word_to_idx = [x for x in r if x in word_to_idx]
            if len(l_in_word_to_idx)/len(l) > 0.85 and\
               len(r_in_word_to_idx)/len(r) > 0.85:
                left_idx.append([word_to_idx[x] for x in l_in_word_to_idx])
                right_idx.append([word_to_idx[x] for x in r_in_word_to_idx])
                f_sim.append(s)
    if shuffle:
        shuffled_zip = list(zip(left_idx, right_idx, f_sim))
        random.shuffle(shuffled_zip)
        unzipped_shuffle = list(zip(*shuffled_zip))
        return list(unzipped_shuffle[0]), list(unzipped_shuffle[1]), list(unzipped_shuffle[2])
    else:
        return left_idx, right_idx, f_sim


def evaluate(model, left_idx, right_idx, sim, dataset):
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
    cos = 1 - spatial.distance.cdist(left_vectors, right_vectors, 'cosine')
    diag = np.diagonal(cos)
    spr = _spearman(sim, diag)
    return spr


def evaluate_distributional_space(model, vocab_filepath, dataset):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Checking embeddings quality against {} similarity ratings'
                .format(dataset))
    left_idx, right_idx, sim = load_words_and_sim_(vocab_filepath, dataset)
    spr = evaluate(model, left_idx, right_idx, sim, dataset)
    # logger.info('Cosine distribution stats on MEN:')
    # logger.info('   Min = {}'.format(diag.min()))
    # logger.info('   Max = {}'.format(diag.max()))
    # logger.info('   Average = {}'.format(diag.mean()))
    # logger.info('   Median = {}'.format(np.median(diag)))
    # logger.info('   STD = {}'.format(np.std(diag)))
    logger.info('SPEARMAN: {}'.format(spr))
    return spr


def _load_kfold_train_test_dict(left_idx, right_idx, sim, kfold_size):
    kfold_dict = {}
    len_test_set = max(math.floor(len(sim)*kfold_size), 1)
    fold = 1
    max_num_fold = math.floor(len(sim) / len_test_set)
    while fold <= max_num_fold:
        test_start_idx = (fold-1)*len_test_set
        test_end_len = test_start_idx + len_test_set
        test_end_idx = test_end_len if test_end_len <= len(sim) else len(sim)
        test_set_idx_set = set(range(test_start_idx, test_end_idx))
        train_set_idx_set = set(range(0, len(sim))) - test_set_idx_set
        train_set_idx_list = sorted(list(train_set_idx_set))
        kfold_dict[fold] = {
            'train': {
                'left_idx': [left_idx[idx] for idx in train_set_idx_list],
                'right_idx': [right_idx[idx] for idx in train_set_idx_list],
                'sim': [sim[idx] for idx in train_set_idx_list]
            },
            'test': {
                'left_idx': left_idx[test_start_idx:test_end_idx],
                'right_idx': right_idx[test_start_idx:test_end_idx],
                'sim': sim[test_start_idx:test_end_idx]
            }
        }
        fold += 1
    return kfold_dict


def load_kfold_train_test_dict(vocab_filepath, dataset, kfold_size):
    """Return a kfold train/test dict.

    The dict has the form dict[kfold_num] = {train_dict, test_dict} where
    train_dict = {left_idx, right_idx, sim}
    the train/test division follows kfold_size expressed as a precentage
    of the total dataset dedicated to testing.
    kfold_size should be a float > 0 and <= 0.5
    """
    left_idx, right_idx, sim = load_words_and_sim(vocab_filepath, dataset,
                                                  shuffle=True)
    print(dataset)
    print(len(sim))
    return _load_kfold_train_test_dict(left_idx, right_idx, sim, kfold_size)
