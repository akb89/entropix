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

import entropix.core.reducer as reducer
import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('compute_entropy', 'compute_pairwise_cosine_sim',
           'compute_singvectors_distribution')


def compute_energy(singvalues_filepath, dims_filepath):

    dims = dutils.load_dimensions_list(dims_filepath)
    singvalues = np.load(singvalues_filepath)
    singvalues = reducer._get_sorted_singvalues(singvalues)
    total_energy = np.sum(singvalues**2)
    curr_energy = 0
    for d in dims:
        curr_energy += singvalues[d]**2
    return curr_energy/total_energy


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


def compute_singvectors_distribution(output_dirpath, model, save_figs):
    """Compute IPR and entropy metrics on singular vectors"""
    output_filepath = futils.get_singvectors_distribution_filepath(output_dirpath, model)
    Umatrix_filepath = futils.get_singvectors_filepath(model)
    Dmatrix_filepath = futils.get_singvalues_filepath(model)

    logger.info('Loading singular values from {}'.format(Dmatrix_filepath))
    sing_values = np.load(Dmatrix_filepath)
    logger.info('Loading singular vectors from {}'.format(Umatrix_filepath))
    sing_vectors = np.load(Umatrix_filepath)

    logger.info('Computing entropy')
    lam_list = []
    entropy_list = []
    for lam, column in tqdm(zip(sing_values, sing_vectors.T)):
        distribution, bins = np.histogram(column, bins='auto')

        # if save_figs:
        #     fig = plt.figure()
        #     ax = plt.subplot(111)
        #     ax.bar(bins[:-1], distribution, 0.02)
        #     plt.title('{}'.format(lam))
        #     fig.savefig('{}/distribution_{}.png'.format(output_dirpath, lam))

        N = len(column)
        normalized_distribution = [x/N for x in distribution if x > 0]

        entropy = 0
        for value in tqdm(normalized_distribution):
            entropy -= value*math.log2(value)

        entropy_list.append(entropy)
        lam_list.append(lam)

    logger.info('Writing results to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        print('lambda_i\tH(u_i)', file=output_stream)
        for lam, h in zip(lam_list, entropy_list):
            print('{}\t{}'.format(lam, h), file=output_stream)

    return entropy_list
