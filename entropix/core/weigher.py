"""Compute ppmi weighing on a sparse matrix."""
import logging
import collections
import math
import scipy.sparse as sparse
import entropix.utils.files as futils

from tqdm import tqdm

logger = logging.getLogger(__name__)

__all__ = ('weigh_ppmi')

def _load_model(model_filepath):
    return sparse.load_npz(model_filepath)


def _load_vocabulary(model_filepath):
    vocab_filepath = futils.get_vocab_filepath(model_filepath)
    idx_to_word_dic = {}
    with open(vocab_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            idx_to_word_dic[int(linesplit[0])] = linesplit[1]
    return idx_to_word_dic


def _load_counts(counts_filepath):
    word_counts = {}
    with open(counts_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            word_counts[linesplit[0]] = int(linesplit[1])
    return word_counts


def ppmi(csr_matrix, idx_to_word_dic, counts):

    N = sum(counts.values())
    rows, cols = csr_matrix.nonzero()

    sorted_indexes = sorted(zip(rows, cols))
    rows_to_cols = collections.defaultdict(list)
    for row, col in sorted_indexes:
        rows_to_cols[row].append(col)

    weighted_rows = []
    weighted_cols = []
    weighted_data = []

    lastrow = None
    for row in tqdm(rows_to_cols):
        if lastrow is None or not row == lastrow:
            matrix_row = csr_matrix.getrow(row).toarray()[0]
            lastrow = row
#        print(matrix_row[0])
#        input()
        f_row = counts[idx_to_word_dic[row]]
        for col in tqdm(rows_to_cols[row]):
            f_col = counts[idx_to_word_dic[col]]
            raw_count = matrix_row[col]
            pmi = math.log2(raw_count*N/(f_row*f_col))
            if pmi > 0:
                weighted_rows.append(row)
                weighted_cols.append(col)
                weighted_data.append(pmi)
#            if pmi < 0:
#                print('computing ppmi between {} and {}: '
#                      'frequency of {} is {} and frequency of {} is {}, '
#                      'co-occurrences counts: {} '
#                      'ppmi: {}'.format(idx_to_word_dic[row], idx_to_word_dic[col], idx_to_word_dic[row], f_row, idx_to_word_dic[col], f_col, raw_count, pmi))
#                input()
    M = sparse.csr_matrix((weighted_data, (weighted_rows, weighted_cols)),
                          shape=(len(idx_to_word_dic), len(idx_to_word_dic)),
                          dtype='f')
    return M



def weigh(output_dirpath, model_filepath, counts_filepath, selected_function):

    output_filepath_weighted_matrix =\
     futils.get_weightedmatrix_filepath(output_dirpath, model_filepath)
    M = _load_model(model_filepath)
    idx_to_word_dic = _load_vocabulary(model_filepath)
    counts = _load_counts(counts_filepath)

    weighted_M = selected_function(M, idx_to_word_dic, counts)
    sparse.save_npz(output_filepath_weighted_matrix, weighted_M)
