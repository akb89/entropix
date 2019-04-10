"""Data utils."""

import os
import logging
from scipy import sparse

logger = logging.getLogger(__name__)

__all__ = ('load_model_from_npz', 'load_vocab_mapping')


def load_model_from_npz(model_filepath):
    """Load sparse scipy matrix from .npz filepath"""
    return sparse.load_npz(model_filepath)


def load_vocab_mapping(vocab_filepath):
    """Load mapping between vocabulary and index from .vocab filepath"""
    idx_to_word_dic = {}
    with open(vocab_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            idx_to_word_dic[int(linesplit[0])] = linesplit[1]
    return idx_to_word_dic


def load_words_set(wordlist_filepath):
    """Load words from a file into a set"""
    words = set()
    with open(wordlist_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            words.add(line.strip().lower())
    return words


def load_dimensions_list(dimensions_filepath):
    ret = []
    with open(dimensions_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linestrip = line.strip()
            ret.append(int(linestrip))
    return ret
