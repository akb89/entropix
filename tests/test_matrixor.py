import numpy as np

import entropix.core.matrixor as matrixor
import entropix.utils.metrix as metrix


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


def test_ao_rotation():
    A = np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]])
    B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    BR = matrixor.apply_ao_rotation(A, B)
    assert abs(metrix.cosine_similarity(B[0], B[1]) - metrix.cosine_similarity(BR[0], BR[1])) < 0.000001
    assert abs(metrix.cosine_similarity(B[1], B[2]) - metrix.cosine_similarity(BR[1], BR[2])) < 0.000001
    AR = matrixor.apply_ao_rotation(B, A)
    assert abs(metrix.cosine_similarity(A[0], A[1]) - metrix.cosine_similarity(AR[0], AR[1])) < 0.000001
    assert abs(metrix.cosine_similarity(A[1], A[2]) - metrix.cosine_similarity(AR[1], AR[2])) < 0.000001


def test_apply_ao_with_scaling():
    A = np.array([[0, 1, 0], [1, 1, 1], [1, 1, 0]])
    B = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    T = matrixor.apply_absolute_orientation_with_scaling(A, B)
    assert abs(metrix.cosine_similarity(B[0], B[1]) - metrix.cosine_similarity(T[0], T[1])) < 0.000001
    assert abs(metrix.cosine_similarity(B[1], B[2]) - metrix.cosine_similarity(T[1], T[2])) < 0.000001
    U = matrixor.apply_absolute_orientation_with_scaling(B, A)
    assert abs(metrix.cosine_similarity(A[0], A[1]) - metrix.cosine_similarity(U[0], U[1])) < 0.000001
    assert abs(metrix.cosine_similarity(A[1], A[2]) - metrix.cosine_similarity(U[1], U[2])) < 0.000001
    assert abs(metrix.root_mean_square_error(A, T) - metrix.root_mean_square_error(U, B)) < 0.000001
