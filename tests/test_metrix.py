"""Test utils.metrix."""

import numpy as np
import scipy.spatial as spatial
import entropix.utils.metrix as metrix


def test_ppmi():
    matrix1 = np.array([[0, 1, 2, 3], [0, 1, 1, 1]], dtype='f')
    matrix2 = np.array([[1, 0, 0, 0], [0, 1, 2, 3]], dtype='f')
    assert matrix1.shape == (2, 4)
    assert matrix1.shape[1] == matrix2.shape[1] == 4
    #normalized_matrix1 = matrix1 / np.apply_along_axis(np.linalg.norm, 1, matrix1).reshape(-1, 1)
    #normalized_matrix2 = matrix2 / np.apply_along_axis(np.linalg.norm, 1, matrix2).reshape(-1, 1)
    # np.testing.assert_array_almost_equal(
    #     normalized_matrix1, np.array(
    #         [[0, 0.267261241912424, 0.534522483824849, 0.801783725737273],
    #          [0, 0.577350269189626, 0.577350269189626, 0.577350269189626]]))
    # np.testing.assert_array_almost_equal(
    #     normalized_matrix2, np.array(
    #         [[1, 0, 0, 0],
    #          [0, 0.267261241912424, 0.534522483824849, 0.801783725737273]]))
    # euc = spatial.distance.cdist(normalized_matrix1, normalized_matrix2, 'euclidean')
    # diag = np.diagonal(euc)
    # np.testing.assert_array_almost_equal(diag, np.array([1.4142135623731, 0.385175]))
    # euc_sim = 1 - euc / matrix1.shape[1]
    # euc_sim_diag = np.diag(euc_sim)
    # sim = metrix.similarity(matrix1, matrix2, distance='euclidean')
    # np.testing.assert_array_almost_equal(euc_sim_diag, sim)
    euc = spatial.distance.cdist(matrix1, matrix2, 'euclidean')
    diag = np.diag(euc)
    np.testing.assert_array_almost_equal(
        diag, np.array(
            [3.87298334620742, 2.23606797749979]))
    normalized = (diag - np.min(diag)) / (np.max(diag) - np.min(diag))
    sim = metrix.similarity(matrix1, matrix2, distance='euclidean')
    np.testing.assert_array_almost_equal(sim, normalized)
