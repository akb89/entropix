"""Compute ppmi weighing on a sparse matrix."""
import logging
import math
import scipy.sparse as sparse
import entropix.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('weigh_ppmi')


def ppmi(csr_matrix):
    """Return a ppmi-weighted CSR sparse matrix from an input CSR matrix."""
    logger.info('Weighing raw count CSR matrix via PPMI')
    weighted_rows = []
    weighted_cols = []
    weighted_data = []
    matrix = csr_matrix.tocoo()
    words_counts = csr_matrix.sum(axis=1)
    total_count = words_counts.sum()
    logger.info('Iterating over CSR matrix...')
    for row, col, value in zip(matrix.row, matrix.col, matrix.data):
        x_count = words_counts[row]
        y_count = words_counts[col]
        pmi = math.log2(value*total_count/(x_count*y_count))
        if pmi > 0:
            weighted_rows.append(row)
            weighted_cols.append(col)
            weighted_data.append(pmi)
    logger.info('Creating sparse PPMI CSR matrix...')
    return sparse.csr_matrix((weighted_data, (weighted_rows, weighted_cols)),
                             shape=csr_matrix.shape,
                             dtype='f')


def weigh(output_dirpath, model_filepath, weighing_func):
    output_filepath_weighted_matrix =\
     futils.get_weightedmatrix_filepath(output_dirpath, model_filepath)
    M = sparse.load_npz(model_filepath)
    if weighing_func == 'ppmi':
        weighted_M = ppmi(M)
        sparse.save_npz(output_filepath_weighted_matrix, weighted_M)
    else:
        raise Exception('Unsupported weighing-func = {}'.format(weighing_func))
