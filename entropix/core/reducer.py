"""SVD."""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

__all__ = ('apply_svd', 'reduce')


def reduce(singvalues, singvectors, alpha):
    """Return U*S^alpha."""
    pass


def apply_svd(model_filepath, dim, sing_values_filepath,
              sing_vectors_filepath, compact=False):
    """Apply SVD to matrix and save singular values and vectors to files.

    If compact is true, only non-null singular values will be kept.
    """
    M = sparse.load_npz(model_filepath)
    if dim == 0 or dim >= M.shape[1]:
        dim = M.shape[1] - 1
    logger.info('Applying SVD on sparse matrix with k = {}'.format(dim))
    U, S, _ = svds(M, k=dim, which='LM', return_singular_vectors='u')
    if compact:
        if np.all(S):
            raise Exception('Compact option set to true but all singular '
                            'values are non-null. Try using reduce with '
                            'a higher dim parameter')
        logger.info('Compact set to true: keeping non-null singular values '
                    'only')
        first_zero_rank = np.nonzero(S)[0].size
        S = S[:first_zero_rank]
        U = U[:, :first_zero_rank]
    logger.info('Saving singular values to {}'.format(sing_values_filepath))
    np.save(sing_values_filepath, S)
    logger.info('Saving singular vectors to {}'.format(sing_vectors_filepath))
    np.save(sing_vectors_filepath, U)
    return U, S
