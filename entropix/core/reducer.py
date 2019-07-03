"""SVD."""

import logging
import joblib
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd
from sklearn.decomposition import FastICA

import entropix.utils.data as dutils
import entropix.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('apply_svd', 'reduce', 'apply_fast_ica')


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


def _get_reduced_sparse_matrix(sparse_matrix_filepath, dataset,
                               vocab_filepath):
    logger.info('Reducing sparse matrix to {} dataset from {}'
                .format(dataset, sparse_matrix_filepath))
    logger.info('Loading sparse matrix from {}'.format(sparse_matrix_filepath))
    sparse_matrix = sparse.load_npz(sparse_matrix_filepath)
    logger.info('Reducing sparse matrix of shape = {}'
                .format(sparse_matrix.shape))
    vocab = dutils.load_vocab(vocab_filepath)
    left_idx, right_idx, _ = dutils.load_dataset(dataset, vocab)
    dataset_idx = list(set(left_idx) | set(right_idx))
    fit_vocab_filepath = futils.get_fit_vocab_filepath(
        vocab_filepath, dataset)
    dutils.save_fit_vocab(dataset_idx, vocab, fit_vocab_filepath)
    return sparse_matrix[dataset_idx]


def reduce(sparse_matrix_filepath, dataset, vocab_filepath):
    """Reducing a sparse matrix to the rows of the words contained in a dataset.

    Converting the sparse reduced matrix to a dense one after removing all
    zero columns.
    """
    reduced_sparse = _get_reduced_sparse_matrix(
        sparse_matrix_filepath, dataset, vocab_filepath)
    logger.info('Reduced to sparse matrix of shape = {}'.format(reduced_sparse.shape))
    nonzero_columns = sorted(set(reduced_sparse.nonzero()[1]))
    reduced_sparse_nonzero = reduced_sparse[:, nonzero_columns]
    logger.info('Converting to dense nonzero (columns) matrix of shape = {}'
                .format(reduced_sparse_nonzero.shape))
    output_dense_mtx_path = futils.get_dense_mtx_filepath(sparse_matrix_filepath, dataset)
    logger.info('Saving output dense matrix to: {}'.format(output_dense_mtx_path))
    np.save(output_dense_mtx_path, reduced_sparse_nonzero.todense())


def _apply_sparse_svd(sparse_matrix_filepath, dim, sing_values_filepath,
                      sing_vectors_filepath, which, dataset=None,
                      vocab_filepath=None, compact=False):
    if dataset:
        logger.info('Applying SVD on sparse matrix limited to word pairs '
                    'found in {}, with which = {} and k = {}'
                    .format(dataset, which, dim))
    else:
        logger.info('Applying SVD on sparse matrix with which = {} and k = {}'
                    .format(which, dim))
    if dataset and not vocab_filepath:
        raise Exception('--vocab parameter is required when specifying '
                        'on entropix svd --dataset')
    if dataset:
        M = _get_reduced_sparse_matrix(sparse_matrix_filepath, dataset,
                                       vocab_filepath)
    else:
        M = sparse.load_npz(sparse_matrix_filepath)
    if dim == 0 or dim >= min(M.shape):
        dim = min(M.shape) - 1
    logger.info('Applying SVD...')
    U, S, _ = svds(M, k=dim, which=which, return_singular_vectors='u')
    if compact:
        if np.all(S):
            logger.warning('Compact option set to true but all singular '
                           'values are non-null. Try using svd with '
                           'a higher dim parameter')
        else:
            logger.info('Compact set to true: keeping non-null singular values '
                        'only')
            first_zero_rank = np.nonzero(S)[0].size
            S = S[:first_zero_rank]
            U = U[:, :first_zero_rank]
    orig_nrj = M.power(2).sum()
    reduced_nrj = np.sum(S ** 2)
    logger.info('Energy of original matrix = {}'.format(orig_nrj))
    logger.info('Energy of reduced matrix = {}'.format(reduced_nrj))
    logger.info('Energy ration = {}%'.format((reduced_nrj/orig_nrj)*100))
    logger.info('Saving singular values of shape = {} to {}'
                .format(S.shape, sing_values_filepath))
    S = S[::-1]  # put singular values in decreasing order of values
    np.save(sing_values_filepath, S)
    logger.info('Saving singular vectors of shape = {} to {}'
                .format(U.shape, sing_vectors_filepath))
    U = U[:, ::-1]  # put singular vectors in decreasing order of sing. values
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
              sing_vectors_filepath, which, dataset=None,
              vocab_filepath=None, compact=False):
    """Apply SVD to matrix and save singular values and vectors to files.

    If compact is true, only non-null singular values will be kept.
    """
    # if model_filepath.endswith('.npz'):
    #     M = sparse.load_npz(model_filepath)
    # elif model_filepath.endswith('.npy'):
    #     DM = np.load(model_filepath)
    #     M = sparse.csr_matrix(DM)
    # else:
    #     raise Exception('Unsupported model extension. Should be .npz or .npy: '
    #                     '{}'.format(model_filepath))
    _apply_sparse_svd(model_filepath, dim, sing_values_filepath,
                      sing_vectors_filepath, which, dataset, vocab_filepath,
                      compact)


def apply_fast_ica(model_filepath, dataset, vocab_filepath):
    model = _get_reduced_sparse_matrix(model_filepath, dataset, vocab_filepath)
    X = model.todense()
    logger.info('Running FastICA on {} components...'.format(model.shape[0]))
    # X = sparse.load_npz(model_filepath)
    transformer = FastICA(n_components=model.shape[0], max_iter=1000, whiten=True)
    X_transformed = transformer.fit_transform(X)
    ica_model_filepath = futils.get_ica_model_filepath(model_filepath, dataset)
    logger.info('Saving output ICA model to {}'.format(ica_model_filepath))
    joblib.dump(X_transformed, ica_model_filepath, compress=0)
