import numpy as np

import entropix.core.matrixor as matrixor


def test_center():
    matrix = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
    centered = matrixor.center(matrix)
    np.testing.assert_array_equal(
        centered, [[-1, -1], [1, -1], [-1, 1], [1, 1]])


def test_sum_outer_product():
    A = np.array([[0, 1, 0]])
    B = np.array([[1, 1, 0]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    A = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    B = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 2, 1], [1, 2, 1], [1, 2, 1]])


def test_sum_inner_product():
    A = np.array([[0, 1, 0]])
    B = np.array([[1, 1, 0]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 1
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 2
    A = np.array([[0, 1, 0], [1, 0, 1], [1, 1, 1]])
    B = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    c = matrixor.compute_sum_inner_product(A, B)
    assert c == 4
