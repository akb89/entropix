"""SVD."""

import logging

from scipy import sparse
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

__all__ = ('reduce_matrix_via_svd')


def reduce_matrix_via_svd(model_filepath, dim, dense_model_filepath,
                          diag_matrix_filepath):
    """Apply SVD to matrix and save singular values and vectors to files."""
    M = sparse.load_npz(model_filepath)
    if dim == 0:
        dim = M.shape[1] - 1
    U, S, Vt = svds(M, k=dim)
    logger.info('Saving singular values to {}'.format(diag_matrix_filepath))
    sparse.save_npz(diag_matrix_filepath, S)
    logger.info('Saving singular vectors to {}'.format(dense_model_filepath))
    sparse.save_npz(dense_model_filepath, U)
