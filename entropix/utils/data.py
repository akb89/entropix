"""Data utils."""
import logging
import math
import random
from collections import defaultdict

import embeddix

logger = logging.getLogger(__name__)

__all__ = ('load_splits_dict')


def _load_splits_dict(left_idx, right_idx, sim, kfold_size):
    kfold_dict = defaultdict(defaultdict)
    len_test_set = max(math.floor(len(sim)*kfold_size), 1)
    fold = 1
    max_num_fold = math.floor(len(sim) / len_test_set)
    while fold <= max_num_fold:
        test_start_idx = (fold-1)*len_test_set
        test_end_len = test_start_idx + len_test_set
        test_end_idx = test_end_len if test_end_len <= len(sim) else len(sim)
        test_split_idx_set = set(range(test_start_idx, test_end_idx))
        kfold_dict[fold]['test'] = {
            'left_idx': left_idx[test_start_idx:test_end_idx],
            'right_idx': right_idx[test_start_idx:test_end_idx],
            'sim': sim[test_start_idx:test_end_idx]
        }
        train_split_idx_set = set(range(0, len(sim))) - test_split_idx_set
        train_split_idx_list = sorted(list(train_split_idx_set))
        kfold_dict[fold]['train'] = {
            'left_idx': [left_idx[idx] for idx in train_split_idx_list],
            'right_idx': [right_idx[idx] for idx in train_split_idx_list],
            'sim': [sim[idx] for idx in train_split_idx_list]
        }
        fold += 1
    return kfold_dict


def load_splits_dict(dataset, vocab, kfold_size=0):
    """Return a kfold train/test dict.

    The dict has the form dict[kfold_num] = {train_dict, test_dict} where
    train/test_dict = {left_idx, right_idx, sim}
    the train/test division follows kfold_size expressed as a precentage
    of the total dataset dedicated to testing.
    kfold_size should be a float >= 0 and <= 0.5
    """
    left_idx, right_idx, f_sim = embeddix.load_dataset(dataset, vocab)
    if kfold_size == 0:
        return {
            1: {
                'train': {
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'sim': f_sim
                },
                'test': {
                    'left_idx': left_idx,
                    'right_idx': right_idx,
                    'sim': f_sim
                }
            }
        }
    # shuffle everything before converting to kfold
    shuffled_zip = list(zip(left_idx, right_idx, f_sim))
    random.shuffle(shuffled_zip)
    unz_shuffle = list(zip(*shuffled_zip))
    left_idx = list(unz_shuffle[0])
    right_idx = list(unz_shuffle[1])
    f_sim = list(unz_shuffle[2])
    return _load_splits_dict(left_idx, right_idx, f_sim, kfold_size)
