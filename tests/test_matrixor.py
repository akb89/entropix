import numpy as np

import entropix.core.matrixor as matrixor


def test_center():
    matrix = np.array([[1, 1], [3, 1], [1, 3], [3, 3]])
    centered = matrixor.center(matrix)
    np.testing.assert_array_equal(
        centered, [[-1, -1], [1, -1], [-1, 1], [1, 1]])


def test_sum_outer_product():
    A = np.array([[0, 1, 0], [1, 0, 1]])
    B = np.array([[0, 0, 0], [1, 1, 1]])
    C = matrixor.compute_sum_outer_product(A, B)
    np.testing.assert_array_equal(
        C, [[1, 0, 1], [1, 0, 1], [1, 0, 1]])
