"""SVD."""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)

__all__ = ('apply_svd', 'reduce')


def _get_reduced_rank(singvalues, energy_ratio):
    """Return the last rank to keep energy% of the original energy."""
    total_energy = np.sum(singvalues**2)
    reduced_energy = (energy_ratio/100) * total_energy
    for idx, _ in enumerate(singvalues):
        if np.sum(singvalues[idx:]**2) < reduced_energy:
            return idx-1
    return len(singvalues)-1


def _sort_values(singvalues, singvectors):
    if singvalues[0] <= singvalues[1]:
        return singvalues, singvectors
    return singvalues[::-1], singvectors[:, ::-1]


def reduce(singvalues, singvectors, alpha, energy, output_filepath=None):
    """Return U*S^alpha."""
    logger.info('Reducing singular vectors and values...')
    if not (0 <= energy <= 100):
        raise Exception('Invalid energy value: {}. Should be in [0, 100]'
                        .format(energy))
    if not (0 <= alpha <= 1):
        raise Exception('Invalid alpha value: {}. Should be in [0, 1]'
                        .format(alpha))
    # Make sure that singvalues/singvectors are ranked in incr. order of val
    singvalues, singvectors = _sort_values(singvalues, singvectors)
    if energy < 100:
        reduced_energy_rank = _get_reduced_rank(singvalues, energy)
        singvalues = singvalues[reduced_energy_rank:]
        singvectors = singvectors[:, reduced_energy_rank:]
    singvalues = np.diag(singvalues)
    if alpha == 1:
        reduced = np.matmul(singvectors, singvalues)
    elif 0 < alpha < 1:
        reduced = np.matmul(singvectors, np.power(singvalues, alpha))
    else:
        reduced = singvectors
    if output_filepath:
        np.save(output_filepath, reduced)
    return reduced


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
                            'values are non-null. Try using svd with '
                            'a higher dim parameter')
        logger.info('Compact set to true: keeping non-null singular values '
                    'only')
        first_zero_rank = np.nonzero(S)[0].size
        S = S[:first_zero_rank]
        U = U[:, :first_zero_rank]
    logger.info('Energy of original matrix = {}'.format(M.power(2).sum()))
    logger.info('Energy of reduced matrix = {}'.format(np.sum(S ** 2)))
    logger.info('Saving singular values to {}'.format(sing_values_filepath))
    np.save(sing_values_filepath, S)
    logger.info('Saving singular vectors to {}'.format(sing_vectors_filepath))
    np.save(sing_vectors_filepath, U)
    return U, S
