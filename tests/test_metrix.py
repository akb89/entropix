"""Test utils.metrix."""

import numpy as np
import scipy.spatial as spatial
import entropix.utils.metrix as metrix


def test_similarity():
    matrix1 = np.array([[0, 1, 2, 3], [0, 1, 1, 1]], dtype='f')
    matrix2 = np.array([[1, 0, 0, 0], [0, 1, 2, 3]], dtype='f')
    assert matrix1.shape == (2, 4)
    assert matrix1.shape[1] == matrix2.shape[1] == 4
    euc = spatial.distance.cdist(matrix1, matrix2, 'euclidean')
    diag = np.diag(euc)
    np.testing.assert_array_almost_equal(
        diag, np.array(
            [3.87298334620742, 2.23606797749979]))
    normalized = (diag - np.min(diag)) / (np.max(diag) - np.min(diag))
    sim = metrix.similarity(matrix1, matrix2, distance='euclidean')
    np.testing.assert_array_almost_equal(sim, normalized)
    simsim = 1 - spatial.distance.cdist(matrix1, matrix2, 'cosine')
    sim_cos = np.diagonal(simsim)
    np.testing.assert_array_almost_equal(metrix.similarity(matrix1, matrix2,
                                                           distance='cosine'),
                                         sim_cos)
