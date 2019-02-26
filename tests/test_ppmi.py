"""Test PPMI weigher code."""

import numpy as np
import scipy.sparse as sparse
import entropix.core.weigher as weigher


def test_ppmi():
    matx = np.array([[2, 4, 3, 8], [2, 3, 2, 3], [3, 2, 7, 6], [8, 3, 5, 1]], dtype='f')
    csr_matrix = sparse.csr_matrix(matx)
    words = csr_matrix.sum(axis=1)
    contexts = csr_matrix.sum(axis=0)
    np.testing.assert_array_equal(words,
                                  np.array([[17], [10], [18], [17]]))
    np.testing.assert_array_equal(contexts,
                                  np.array([[15, 12, 17, 18]]))
    x = np.array([[0.11764706, 0.23529412, 0.17647059, 0.47058824],
                  [0.2,        0.3,        0.2,        0.3       ],
                  [0.16666667, 0.11111111, 0.38888889, 0.33333333],
                  [0.47058824, 0.17647059, 0.29411765, 0.05882353]], dtype='f')
    y = np.array([[0.13333334, 0.33333334, 0.1764706,  0.44444445],
                  [0.13333334, 0.25,       0.11764706, 0.16666667],
                  [0.2,        0.16666667, 0.4117647,  0.33333334],
                  [0.53333336, 0.25,       0.29411766, 0.05555556]], dtype='f')
    xy = np.array([[0.48627454, 1.2156863,  0.6435986,  1.620915  ],
                   [0.8266668,  1.5500002,  0.7294118,  1.0333334 ],
                   [0.68888897, 0.5740741,  1.4183006,  1.1481482 ],
                   [1.9450982,  0.91176474, 1.0726645,  0.20261438]], dtype='f')
    xy_log = np.array(
        [[-1.0401571,   0.28177103, -0.63576686,  0.6968085 ],
         [-0.2746222,   0.63226837, -0.4551946,   0.04730584],
         [-0.5376566,  -0.8006911,   0.5041634,   0.19930884],
         [ 0.959843,   -0.13326648,  0.10119891, -2.3031914 ]], dtype='f')
    xy_pos_log = np.array(
        [[0.,   0.28177103, 0.,  0.6968085 ],
         [0.,   0.63226837, 0.,   0.04730584],
         [0.,  0.,   0.5041634,   0.19930884],
         [0.959843,   0.,  0.10119891, 0. ]], dtype='f')
    inv_words = sparse.csr_matrix(words).power(-1)
    inv_contexts = sparse.csr_matrix(contexts).power(-1)
    x_test = csr_matrix.multiply(inv_words)
    np.testing.assert_array_almost_equal(np.asarray(x_test.todense()), x)
    y_test = csr_matrix.multiply(inv_contexts)
    np.testing.assert_array_almost_equal(np.asarray(y_test.todense()), y)
    total_count = csr_matrix.sum()
    xy_test = csr_matrix.multiply(inv_words).multiply(inv_contexts).multiply(total_count)
    np.testing.assert_array_almost_equal(np.asarray(xy_test.todense()), xy)
    xy_log_test = xy_test
    xy_log_test.data = np.log2(xy_test.data)
    np.testing.assert_array_almost_equal(np.asarray(xy_log_test.todense()), xy_log)
    print(weigher.ppmi(csr_matrix).todense())
    np.testing.assert_array_almost_equal(weigher.ppmi(csr_matrix).todense(), xy_pos_log)
