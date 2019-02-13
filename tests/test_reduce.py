"""Test the reducer."""
import numpy as np
import entropix.core.reducer as reducer


def test_sort_values():
    S = np.array([10, 9, 8, 1])
    U = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    red_s, red_u = reducer._sort_values(S, U)
    SS = np.array([1, 8, 9, 10])
    UU = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])
    np.testing.assert_array_equal(SS, red_s)
    np.testing.assert_array_equal(UU, red_u)
    red_ss, red_uu = reducer._sort_values(SS, UU)
    np.testing.assert_array_equal(SS, red_ss)
    np.testing.assert_array_equal(UU, red_uu)


def test_get_reduced_rank():
    S = np.array([1, 2, 3, 4])
    assert reducer._get_reduced_rank(S, 100) == 0
    assert reducer._get_reduced_rank(S, 0) == 3
    assert reducer._get_reduced_rank(S, 90) == 1
    assert reducer._get_reduced_rank(S, 80) == 2
    assert reducer._get_reduced_rank(S, 50) == 3

def test_reduce():
    S = np.array([1, 2, 3, 4])
    U = np.array([[4, 3, 2, 1], [8, 7, 6, 5]])
    SU1001 = np.array([[4, 6, 6, 4], [8, 14, 18, 20]])
    SU1005 = np.array([[4, 4.242640, 3.464101, 2], [8, 9.899494, 10.392304, 10]])
    SU801 = np.array([[6, 4], [18, 20]])
    SU800 = np.array([[2, 1], [6, 5]])
    np.testing.assert_array_equal(reducer.reduce(S, U, 1, 100), SU1001)
    np.testing.assert_array_equal(reducer.reduce(S, U, 0, 100), U)
    np.testing.assert_array_almost_equal(reducer.reduce(S, U, .5, 100), SU1005)
    np.testing.assert_array_equal(reducer.reduce(S, U, 1, 80), SU801)
    np.testing.assert_array_equal(reducer.reduce(S, U, 0, 80), SU800)
