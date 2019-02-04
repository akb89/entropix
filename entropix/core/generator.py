"""A basic count-based model using sparse matrices and no ppmi."""
import os
import logging
from scipy import sparse

from tqdm import tqdm
import entropix.core.counter as count
import entropix.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('generate_distributional_model')


def generate_distributional_model(output_dirpath, corpus_filepath,
                                  min_count, win_size):
    """Generate a count-based distributional model."""
    output_filepath_matrix = futils.get_sparsematrix_filepath(
                                    output_dirpath, corpus_filepath,
                                    min_count, win_size)
    output_filepath_map = futils.get_vocab_filepath(output_filepath_matrix)

    word_to_count_dic = count.count_words(corpus_filepath=corpus_filepath,
                                          min_count=min_count)
    word_to_idx_dic = {word: idx for idx, word
                       in enumerate(word_to_count_dic.keys())}
    rows = []
    columns = []
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        for line in tqdm(input_stream):
            tokens = line.strip().split()
            for token_pos, token in enumerate(tokens):
                start = 0 if win_size == 0 else max(0, token_pos-win_size)
                while start < token_pos:
                    ctx = tokens[start]
                    if token in word_to_idx_dic and ctx in word_to_idx_dic:
                        token_idx = word_to_idx_dic[token]
                        ctx_idx = word_to_idx_dic[ctx]
                        rows.append(token_idx)
                        columns.append(ctx_idx)
                        rows.append(ctx_idx)
                        columns.append(token_idx)
                    start += 1

    logger.info('Building CSR sparse matrix')
    M = sparse.csr_matrix(([1]*len(rows), (rows, columns)),
                          shape=(len(word_to_idx_dic), len(word_to_idx_dic)),
                          dtype='f')
    logger.info('Matrix info: {} non-zero entres, {} shape, {:.6f} density'
                .format(M.getnnz(), M.shape,
                        M.getnnz()*1.0/(M.shape[0]*M.shape[1])))
    logger.info('Saving matrix to {}.npz'.format(output_filepath_matrix))
    sparse.save_npz(output_filepath_matrix, M)
    with open(output_filepath_map, 'w', encoding='utf-8') as output_stream:
        for word, idx in word_to_idx_dic.items():
            print('{}\t{}'.format(idx, word), file=output_stream)
    return M, word_to_idx_dic
