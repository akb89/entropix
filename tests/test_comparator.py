"""Test the comparator."""
import numpy as np

import entropix.core.comparator as comparator


def test_compare():
    A = np.array([[0,1,2,3], [0,1,2,3], [0,1,2,3], [0,1,2,3]])
    B = np.array([[0,1,2,3], [0,1,2,3], [0,1,2,3], [0,1,2,3]])
    variance = comparator.compare(A, B, {}, {}, n=2, num_threads=1)


def test_get_n_nearest_neighbors():
    A = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [2, 2]])
    n1 = comparator._get_n_nearest_neighbors(0, A, 2)
    assert n1 == {2, 4}
    n2 = comparator._get_n_nearest_neighbors(2, A, 2)
    assert n2 == {0, 4}
    n3 = comparator._get_n_nearest_neighbors(1, A, 2)
    assert n3 == {3, 4}


def test_compare_low_ram():
    A = np.array([[0, 1], [1, 0], [0, 1], [1, 1], [2, 2]])
    var = comparator._compare_low_ram(A, A, 2)
    assert var == [0, 0, 0, 0, 0]
    B = np.array([[1, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    var = comparator._compare_low_ram(A, B, 2)
    assert var == [0, 0, 0, 0, 0]
    C = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    var = comparator._compare_low_ram(B, C, 2)
    assert var == [0.5, 0.5, 0, 0.5, 0.5]
