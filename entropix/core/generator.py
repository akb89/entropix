"""A basic count-based model using sparse matrices and no ppmi."""
import logging
import functools
from collections import defaultdict

import multiprocessing
from scipy import sparse
from tqdm import tqdm

import entropix.core.counter as counter
import entropix.utils.files as futils

from entropix.filtering.informativeness import Informativeness

logger = logging.getLogger(__name__)

__all__ = ('generate_distributional_model')


def _count_lines_in_stream(corpus_filepath):
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        return sum(1 for line in input_stream)


def _count_with_info_filter(word_to_idx_dic, win_size, line):
    data_dic = {}
    test_list = [idx for idx in range(90000000)]
    # tokens = line.strip().split()
    # for token_pos, token in enumerate(tokens):
    #     if token not in word_to_idx_dic:
    #         # print('not in dic = {}'.format(token))
    #         continue
    #     context = tokens[max(0, token_pos-win_size): token_pos] + tokens[token_pos+1: min(len(tokens), token_pos+win_size+1)]
    #     context = tuple([w for w in context if w in info.model.wv.vocab and w in word_to_idx_dic])
    #     filtered_context = info.filter_context_words(context)
    #     # print('tokens = {}'.format(tokens))
    #     # print('target = {}'.format(tokens[token_pos]))
    #     # print('context = {}'.format(context))
    #     # print('filtered = {}'.format(filtered_context))
    #     # print('----------------------------------------')
    #     token_idx = word_to_idx_dic[token]
    #     for ctx in filtered_context:
    #         ctx_idx = word_to_idx_dic[ctx]
    #         if token_idx not in data_dic:
    #             data_dic[token_idx] = {}
    #             data_dic[token_idx][ctx_idx] = 1
    #         elif ctx_idx not in data_dic[token_idx]:
    #             data_dic[token_idx][ctx_idx] = 1
    #         else:
    #             data_dic[token_idx][ctx_idx] += 1
    return data_dic


def _count_raw_no_filter(input_stream, data_dic, win_size, word_to_idx_dic,
                         total_num_lines):
    for line in tqdm(input_stream, total=total_num_lines):
        tokens = line.strip().split()
        for token_pos, token in enumerate(tokens):  # raw count with symmetric matrix
            start = 0 if win_size == 0 else max(0, token_pos-win_size)
            while start < token_pos:
                ctx = tokens[start]
                if token in word_to_idx_dic and ctx in word_to_idx_dic:
                    token_idx = word_to_idx_dic[token]
                    ctx_idx = word_to_idx_dic[ctx]
                    data_dic[token_idx][ctx_idx] += 1
                start += 1
    return data_dic


def generate_distributional_model(output_dirpath, corpus_filepath,
                                  min_count, win_size, with_info,
                                  info_model_path, num_threads):
    """Generate a count-based distributional model.

    If info_model is set, will use the model to filter informative context.
    """
    if with_info and not info_model_path:
        raise Exception('You need to specify --info_model if --with-info is set to true')
    if with_info:
        logger.info('Generating DS model with informativeness on {} threads'
                    .format(num_threads))
        global info  # hack to avoid RAM explosion on multiprocessing
        info = Informativeness(info_model_path)
    output_filepath_matrix = futils.get_sparsematrix_filepath(
        output_dirpath, corpus_filepath, min_count, win_size, with_info)
    output_filepath_map = futils.get_vocab_filepath(output_filepath_matrix)

    word_to_count_dic = counter.count_words(corpus_filepath=corpus_filepath,
                                            min_count=min_count)
    word_to_idx_dic = {word: idx for idx, word
                       in enumerate(word_to_count_dic.keys())}
    data_dic = defaultdict(lambda: defaultdict(int))
    logger.info('Counting lines in corpus...')
    total_num_lines = _count_lines_in_stream(corpus_filepath)
    logger.info('Total number of lines = {}'.format(total_num_lines))
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        if not with_info:
            data_dic = _count_raw_no_filter(input_stream, data_dic, win_size,
                                            word_to_idx_dic, total_num_lines)
        else:  # TODO: make info model global otherwise RAM explodes
            with multiprocessing.Pool(processes=num_threads) as pool:
                _process = functools.partial(_count_with_info_filter,
                                             word_to_idx_dic, win_size)
                for _data_dic in tqdm(pool.imap_unordered(_process,
                                                          input_stream),
                                      total=total_num_lines):
                    for row, columns in _data_dic.items():
                        for col, count in columns.items():
                            data_dic[row][col] += count

    logger.info('Building CSR sparse matrix...')
    rows = []
    columns = []
    data = []
    for row_idx in tqdm(data_dic):
        for col_idx in data_dic[row_idx]:
            rows.append(row_idx)
            columns.append(col_idx)
            data.append(data_dic[row_idx][col_idx])
            if not with_info:  # rely on the fact that the matrix is symmetric
                rows.append(col_idx)
                columns.append(row_idx)
                data.append(data_dic[row_idx][col_idx])

    M = sparse.csr_matrix((data, (rows, columns)),
                          shape=(len(word_to_idx_dic), len(word_to_idx_dic)),
                          dtype='f')
    logger.info('Matrix info: {} non-zero entries, {} shape, {:.6f} density'
                .format(M.getnnz(), M.shape,
                        M.getnnz()*1.0/(M.shape[0]*M.shape[1])))
    logger.info('Saving matrix to {}'.format(output_filepath_matrix))
    sparse.save_npz(output_filepath_matrix, M)
    with open(output_filepath_map, 'w', encoding='utf-8') as output_stream:
        for word, idx in word_to_idx_dic.items():
            print('{}\t{}\t{}'.format(idx, word, word_to_count_dic[word]),
                  file=output_stream)
    return M, word_to_idx_dic
