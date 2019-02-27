"""Data utils."""

import os
import logging
import collections
import math
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

def load_index_set(indexlist_filepath):
    """Load indexes (int) from a file into a set"""
    idxs = set()
    with open(indexlist_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            idxs.add(int(line.strip()))
    return idxs


def load_2d_array(input_filepath, index_set, symm=False):
    indexes = set()
    matrix = collections.defaultdict(lambda: collections.defaultdict(float))
    with open(input_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            idx1, idx2, value = line.strip().split()
            idx1, idx2 = int(idx1), int(idx2)
            value = math.log2(float(value)+1)

            if not index_set or (idx1 in index_set and idx2 in index_set):
                indexes.add(idx1)
                indexes.add(idx2)
                matrix[idx1][idx2] = value
                if symm:
                    matrix[idx2][idx1] = value

    return [[matrix[idx1][idx2] for idx2 in indexes] for idx1 in indexes]


def load_2columns(input_filepath):
    x = []
    y = []
    with open(input_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split()
            x.append(float(linesplit[0]))
            y.append(float(linesplit[1]))
    return x, y


def load_intlist(input_filepath):
    ret = []
    with open(input_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            n = int(line.strip())
            ret.append(n)

    return ret


def load_wordlist(input_filepath):
    ret = {}
    with open(input_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            idx = int(linesplit[0])
            wordlist = linesplit[1].split(', ')

            ret[idx] = wordlist

    return ret
