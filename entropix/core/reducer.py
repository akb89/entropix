"""SVD."""

import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd

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


def _get_sorted_singv(singvalues, singvectors):
    if singvalues[0] <= singvalues[1]:
        return singvalues, singvectors
    return singvalues[::-1], singvectors[:, ::-1]


def _get_sorted_singvalues(singvalues):
    if singvalues[0] <= singvalues[1]:
        return singvalues
    return singvalues[::-1]


def _get_top_sorted_singv(singvalues, singvectors, top):
    """Return all but the top-n singvalues/vectors."""
    if not (0 <= top < len(singvalues)):
        raise Exception('Invalid top value: {}. Should be in '
                        '[0, len(singvalues)={}]'.format(top, len(singvalues)))
    sdsingvalues, sdsingvectors = _get_sorted_singv(singvalues, singvectors)
    if top == 0:
        return sdsingvalues, sdsingvectors
    return sdsingvalues[:len(sdsingvalues)-top], sdsingvectors[:, :len(singvalues)-top]


def reduce(singvalues, singvectors, top, alpha, energy, output_filepath=None):
    """Return U*S^alpha.

    Reduce singular values: remove top-n values, get energy*original_energy
    """
    logger.info('Reducing singular vectors and values...')
    if not (0 <= energy <= 100):
        raise Exception('Invalid energy value: {}. Should be in [0, 100]'
                        .format(energy))
    if not (0 <= alpha <= 1):
        raise Exception('Invalid alpha value: {}. Should be in [0, 1]'
                        .format(alpha))
    if not (0 <= top < len(singvalues)):
        raise Exception('Invalid top value: {}. Should be in '
                        '[0, len(singvalues)={}]'.format(top, len(singvalues)))
    # Make sure that singvalues/singvectors are ranked in incr. order of val
    singvalues, singvectors = _get_top_sorted_singv(singvalues, singvectors, top)
    if energy < 100:
        reduced_energy_rank = _get_reduced_rank(singvalues, energy)
        singvalues = singvalues[reduced_energy_rank:]
        singvectors = singvectors[:, reduced_energy_rank:]
    logger.info(singvalues)
    if output_filepath:
        singvalues_filepath = '{}.singvalues.npy'.format(
            output_filepath.split('.npy')[0])
        np.save(singvalues_filepath, singvalues)
    if alpha == 0:
        reduced = singvectors
    elif alpha == 1:
        reduced = np.matmul(singvectors, np.diag(singvalues))
    else:
        reduced = np.matmul(singvectors, np.diag(np.power(singvalues, alpha)))
    if output_filepath:
        np.save(output_filepath, reduced)
    return reduced


def _apply_sparse_svd(M, dim, sing_values_filepath,
                      sing_vectors_filepath, compact=False):
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


def _apply_exact_svd(model_filepath, sing_values_filepath,
                     sing_vectors_filepath):
    logger.info('Applying SVD on dense matrix')
    M = np.load(model_filepath)
    U, S, _ = svd(M)
    logger.info('Saving singular values to {}'.format(sing_values_filepath))
    np.save(sing_values_filepath, S)
    logger.info('Saving singular vectors to {}'.format(sing_vectors_filepath))
    np.save(sing_vectors_filepath, U)


def apply_svd(model_filepath, dim, sing_values_filepath,
              sing_vectors_filepath, compact=False):
    """Apply SVD to matrix and save singular values and vectors to files.

    If compact is true, only non-null singular values will be kept.
    """
    if model_filepath.endswith('.npz'):
        M = sparse.load_npz(model_filepath)
        _apply_sparse_svd(M, dim, sing_values_filepath, sing_vectors_filepath,
                          compact)
    elif model_filepath.endswith('.npy'):
        DM = np.load(model_filepath)
        M = sparse.csr_matrix(DM)
        _apply_sparse_svd(M, dim, sing_values_filepath, sing_vectors_filepath,
                          compact)
    else:
        raise Exception('Unsupported model extension. Should be .npz or .npy: '
                        '{}'.format(model_filepath))
