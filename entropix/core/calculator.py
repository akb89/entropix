"""
Compute entropy metrics. Also compute paiwise cosine similarity between
vocabulary items and their distribution.
"""

import logging
import math
import multiprocessing
import gzip
import functools
import numpy as np
from scipy import sparse
from scipy import spatial
from scipy.stats import shapiro
import matplotlib.pyplot as plt

from tqdm import tqdm

import entropix.utils.files as futils


logger = logging.getLogger(__name__)

__all__ = ('compute_entropy', 'compute_pairwise_cosine_sim',
           'compute_singvectors_distribution')


def compute_entropy(counts):
    """Compute entropy from a counts dict."""
    total_count = sum(counts.values())
    logger.info('Total vocab size = {}'.format(total_count))
    corpus_size = 0
    if total_count > 1e9:
        corpus_size = '{}B'.format(round(total_count / 1e9))
    elif total_count > 1e6:
        corpus_size = '{}M'.format(round(total_count / 1e6))
    elif total_count > 1e3:
        corpus_size = '{}K'.format(round(total_count / 1e3))
    else:
        corpus_size = '{}'.format(round(total_count))
    logger.info('Corpus size = {}'.format(corpus_size))
    vocab_size = 0
    if len(counts) > 1e6:
        vocab_size = '{}M'.format(round(len(counts) / 1e6))
    elif len(counts) > 1e3:
        vocab_size = '{}K'.format(round(len(counts) / 1e3))
    else:
        vocab_size = '{}'.format(round(len(counts)))
    logger.info('Vocab size = {}'.format(vocab_size))
    probs = [count/total_count for count in counts.values()]
    logprobs = [prob * math.log2(prob) for prob in probs]
    entropy = -sum(logprobs)
    logger.info('Entropy = {}'.format(entropy))
    return corpus_size, vocab_size, entropy


def _load_wordlist(wordlist_filepath):
    words = set()
    with open(wordlist_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            words.add(line.strip().lower())
    return words


def _load_vocabulary(vocab_filepath):
    idx_to_word_dic = {}
    with open(vocab_filepath, encoding='utf-8') as input_stream:
        for line in input_stream:
            linesplit = line.strip().split('\t')
            idx_to_word_dic[int(linesplit[0])] = linesplit[1]
    return idx_to_word_dic


def _load_model(model_filepath):
    return sparse.load_npz(model_filepath)


def _process(M, idx_to_word_dic, idx):
    cosines_dic = {}
    vector = M.getrow(idx).toarray()
    for idx2 in idx_to_word_dic:
        if idx2 > idx:
            vector2 = M.getrow(idx2).toarray()
            cosine = 1-spatial.distance.cosine(vector, vector2)
            if not math.isnan(cosine):
                cosines_dic[(idx, idx2)] = cosine
            else:
                logger.info('Undefined cosine similarity for pair '
                            '{} {}'.format(idx_to_word_dic[idx],
                                           idx_to_word_dic[idx2]))
    return cosines_dic, idx


def compute_pairwise_cosine_sim(output_dirpath, model_filepath, vocab_filepath,
                                num_threads, bin_size, wordlist_filepath=None):
    """
    Compute paiwise cosine similarity between vocabulary items.
    The function also computes the distribution of pariwise cosine sim.
    """
    cosinepairs_filepath = futils.get_cosines_filepath(output_dirpath,
                                                       model_filepath)
    distribution_filepath = futils.get_cosines_distribution_filepath(
        output_dirpath, model_filepath)
    number_of_bins = 1/bin_size
    freqdist = [0]*int(number_of_bins)
    words_shortlist = None
    with gzip.open(cosinepairs_filepath, 'wt', encoding='utf-8') as \
      output_cosinepairs, open(distribution_filepath, 'w', encoding='utf-8') \
      as output_distribution:

        if wordlist_filepath:
            words_shortlist = _load_wordlist(wordlist_filepath)

        M = _load_model(model_filepath)
        idx_to_word_dic = _load_vocabulary(vocab_filepath)

        if words_shortlist:
            idx_to_word_dic = {k: w for k, w in idx_to_word_dic.items()
                               if w in words_shortlist}
            logger.info('{} out of {} vocabulary items found.'
                        .format(len(idx_to_word_dic), len(words_shortlist)))

        with multiprocessing.Pool(num_threads) as pool:
            process = functools.partial(_process, M, idx_to_word_dic)
            for partial_cosine_dic, idx in tqdm(pool.imap_unordered(process, idx_to_word_dic.keys())):
                for index_pair, cosine in partial_cosine_dic.items():
                    idx, idx2 = index_pair
                    word1, word2 = idx_to_word_dic[idx], idx_to_word_dic[idx2]
                    freqdist[int(cosine/bin_size)] += 1
                    print('{}\t{}\t{}'
                          .format(word1, word2, cosine), file=output_cosinepairs)

        # write distribution to output file
        print('range\tN', file=output_distribution)
        for i, k in enumerate(freqdist):
            print('{}<=x<{}\t{}'.format(round(i*bin_size, 5),
                                        round((i+1)*bin_size, 5), k),
                  file=output_distribution)

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

        if save_figs:
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.bar(bins[:-1], distribution, 0.02)
            plt.title('{}'.format(lam))
            fig.savefig('{}/distribution_{}.png'.format(output_dirpath, lam))

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
