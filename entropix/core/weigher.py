"""Compute ppmi weighing on a sparse matrix."""
import logging
import numpy as np
import scipy.sparse as sparse
import entropix.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('weigh_ppmi')


def ppmi(csr_matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logger.info('Weighing raw count CSR matrix via PPMI')
    words = sparse.csr_matrix(csr_matrix.sum(axis=1))
    contexts = sparse.csr_matrix(csr_matrix.sum(axis=0))
    total_sum = csr_matrix.sum()
    # csr_matrix = csr_matrix.multiply(words.power(-1)) # #(w, c) / #w
    # csr_matrix = csr_matrix.multiply(contexts.power(-1))  # #(w, c) / (#w * #c)
    # csr_matrix = csr_matrix.multiply(total)  # #(w, c) * D / (#w * #c)
    csr_matrix = csr_matrix.multiply(words.power(-1))\
                           .multiply(contexts.power(-1))\
                           .multiply(total_sum)
    csr_matrix.data = np.log2(csr_matrix.data)  # PMI = log(#(w, c) * D / (#w * #c))
    csr_matrix = csr_matrix.multiply(csr_matrix > 0)  # PPMI
    csr_matrix.eliminate_zeros()
    return csr_matrix


def weigh(output_dirpath, model_filepath, weighing_func):
    output_filepath_weighted_matrix =\
     futils.get_weightedmatrix_filepath(output_dirpath, model_filepath)
    if model_filepath.endswith('.npy'):
        DM = np.load(model_filepath)
        M = sparse.csr_matrix(DM)
    elif model_filepath.endswith('.npz'):
        M = sparse.load_npz(model_filepath)
    else:
        raise Exception('Unsupported model extension. Should be .npz or .npy: '
                        '{}'.format(model_filepath))
    if weighing_func == 'ppmi':
        weighted_M = ppmi(M)
        sparse.save_npz(output_filepath_weighted_matrix, weighted_M)
    else:
        raise Exception('Unsupported weighing-func = {}'.format(weighing_func))
