"""Test the aligner."""
import numpy as np

import entropix.core.aligner as aligner


def test_align_vocab():
    A = np.array([[0], [1], [2], [3], [4], [5]])
    B = np.array([[3], [1], [2], [0], [4]])
    vocabA = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'X': 4, 'Z': 5}
    vocabB = {'D': 0, 'C': 1, 'B': 2, 'A': 3, 'Y': 4}
    C, D, vocab = aligner.align_vocab(A, B, vocabA, vocabB)
    assert C.shape == D.shape
    assert C.shape[0] == 4
    assert C.shape[1] == 1
    assert C[0] == D[0]
    assert C[3] == D[3]
    assert C[1] == D[2]
    assert vocab == {'A': 0, 'B': 1, 'C': 2, 'D': 3}
