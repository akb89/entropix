"""SVD."""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

__all__ = ('reduce_matrix_via_svd')


def reduce_matrix_via_svd(model_filepath, dim, sing_values_filepath,
                          sing_vectors_filepath):
    """Apply SVD to matrix and save singular values and vectors to files."""
    M = sparse.load_npz(model_filepath)
    if dim == 0:
        dim = M.shape[1] - 1
    logger.info('Applying SVD on sparse matrix with k = {}'.format(dim))
    U, S, Vt = svds(M, k=dim)
    logger.info('Saving singular values to {}'.format(sing_values_filepath))
    np.save(sing_values_filepath, S)
    logger.info('Saving singular vectors to {}'.format(sing_vectors_filepath))
    np.save(sing_vectors_filepath, U)
