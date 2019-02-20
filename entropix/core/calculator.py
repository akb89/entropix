"""
Compute entropy metrics. Also compute paiwise cosine similarity between
vocabulary items and their distribution.
"""

import logging
import math
import multiprocessing
import functools
import numpy as np
import scipy
from scipy import sparse
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
#import matplotlib.pyplot as plt

from tqdm import tqdm

import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('compute_entropy', 'compute_pairwise_cosine_sim',
           'compute_singvectors_entropy', 'compute_singvectors_ipr')


def compute_entropy(counts):
    """Compute entropy from a counts dict."""
    total_count = sum(counts.values())
    logger.info('Total corpus size = {}'.format(total_count))
    logger.info('Total vocab size = {}'.format(len(counts)))
    entropy = scipy.stats.entropy(list(counts.values()), base=2)
    logger.info('Entropy = {}'.format(entropy))
    return total_count, len(counts), entropy


def _process(output_dirpath, M, idx_to_word_dic, bin_size, idx):
    number_of_bins = 1/bin_size + 1
    freqdist = [0]*int(number_of_bins)
    vector = M.getrow(idx)
    output_filepath = futils.get_tmp_cosinedist_filepath(output_dirpath, idx)
    similarities = cosine_similarity(vector, M.T, dense_output=False).tocoo()

    for row, col, value in zip(similarities.row, similarities.col, similarities.data):
    #    print(number_of_bins, value, int(value/bin_size))
        freqdist[int(value/bin_size)] += 1

#        print(row, col, value)

#    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
#        for idx2 in idx_to_word_dic:
#            if idx2 > idx:
#                vector2 = M.getrow(idx2)
#                cosine = 1-spatial.distance.cosine(vector, vector2)

#                if not math.isnan(cosine):
#                    freqdist[int(cosine/bin_size)] += 1
#                    word1, word2 = idx_to_word_dic[idx], idx_to_word_dic[idx2]
#                    print('{}\t{}\t{}'.format(word1, word2, cosine), file=output_stream)
#                else:
#                    logger.info('Undefined cosine similarity for pair '
#                                '{} {}'.format(idx_to_word_dic[idx],
#                                               idx_to_word_dic[idx2]))
    return freqdist, idx


def compute_pairwise_cosine_sim(output_dirpath, model_filepath, vocab_filepath,
                                num_threads, bin_size):
    """
    Compute paiwise cosine similarity between vocabulary items.
    The function also computes the distribution of pariwise cosine sim.
    """
    distribution_filepath = futils.get_cosines_distribution_filepath(
        output_dirpath, model_filepath)
    number_of_bins = 1/bin_size + 1
    freqdist = [0]*int(number_of_bins)
    with open(distribution_filepath, 'w', encoding='utf-8') as output_distribution:
        M = dutils.load_model_from_npz(model_filepath)

        similarities = cosine_similarity(M, dense_output=False).tocoo()

        for row_index, col_index, value in zip(similarities.row, similarities.col, similarities.data):
            bin = int(value/bin_size)
            freqdist[bin] +=1
    #     idx_to_word_dic = dutils.load_vocab_mapping(vocab_filepath)
    #     with multiprocessing.Pool(num_threads) as pool:
    #         process = functools.partial(_process, output_dirpath, M,
    #                                      idx_to_word_dic, bin_size)
    #         for partial_freqdist, idx in tqdm(pool.imap_unordered(process, idx_to_word_dic.keys())):
    #             for bin, value in enumerate(partial_freqdist):
    #                 freqdist[bin] += value

        # write distribution to output file

        # print('range\tN', file=output_distribution)
        # for i, k in enumerate(freqdist):
        #     print('{}<=x<{}\t{}'.format(round(i*bin_size, 5),
        #                                 round((i+1)*bin_size, 5), k),
        #           file=output_distribution)

    return freqdist


def compute_singvectors_entropy(output_dirpath, svec_filepath, sval_filepath):
    """Compute entropy metrics on singular vectors"""
    output_filepath = futils.get_singvectors_entropy_filepath(output_dirpath,
                                                              svec_filepath)

    logger.info('Loading singular values from {}'.format(sval_filepath))
    sing_values = np.load(sval_filepath)
    logger.info('Loading singular vectors from {}'.format(svec_filepath))
    sing_vectors = np.load(svec_filepath)

    logger.info('Computing entropy')
    lam_list = []
    entropy_list = []
    for lam, column in tqdm(zip(sing_values, sing_vectors.T)):
        distribution, bins = np.histogram(column)

        N = len(column)
        normalized_distribution = [x/N for x in distribution if x > 0]

        entropy = 0
        for value in normalized_distribution:
            entropy -= value*math.log2(value)

        entropy_list.append(entropy)
        lam_list.append(lam)

    logger.info('Writing results to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        for lam, h in zip(lam_list, entropy_list):
            print('{}\t{}'.format(lam, h), file=output_stream)

    return entropy_list


def compute_singvectors_ipr(output_dirpath, svec_filepath, sval_filepath):
    """Compute IPR metrics on singular vectors"""
    output_filepath = futils.get_singvectors_ipr_filepath(output_dirpath,
                                                          svec_filepath)

    logger.info('Loading singular values from {}'.format(sval_filepath))
    sing_values = np.load(sval_filepath)
    logger.info('Loading singular vectors from {}'.format(svec_filepath))
    sing_vectors = np.load(svec_filepath)

    logger.info('Computing ipr')
    lam_list = []
    ipr_list = []
    for lam, column in tqdm(zip(sing_values, sing_vectors.T)):
        ipr = 0
        for value in column:
            ipr += math.pow(value, 4)

        ipr_list.append(ipr)
        lam_list.append(lam)

    logger.info('Writing results to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        for lam, ipr in zip(lam_list, ipr_list):
            print('{}\t{}'.format(lam, ipr), file=output_stream)

    return ipr_list
