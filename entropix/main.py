"""Welcome to entropix.

This is the entry point of the application.
"""

import os

import math

import argparse
import logging
import logging.config

from collections import defaultdict

import entropix.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)

__all__ = ('count', 'compute')


def compute(counts):
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


def _compute(args):
    logger.info('Computing entropy from file {}'.format(args.counts))
    counts = {}
    with open(args.counts, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            word_count = line.split('\t')
            counts[word_count[0]] = int(word_count[1])
    compute(counts)


def count(output_dirpath, corpus_filepath, min_count=0):
    """Count words in a corpus."""
    if corpus_filepath.endswith('.txt'):
        output_filepath = os.path.join(
            output_dirpath, '{}.counts'.format(
                os.path.basename(corpus_filepath).split('.txt')[0]))
    else:
        output_filepath = os.path.join(
            output_dirpath,
            '{}.counts'.format(os.path.basename(corpus_filepath)))
    _counts = defaultdict(int)
    logger.info('Counting words in {}'.format(corpus_filepath))
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            for word in line.split():
                _counts[word] += 1
    if min_count == 0:
        counts = _counts
    else:
        counts = {word: count for word, count in _counts.items() if count >= min_count}
    logger.info('Saving counts to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        for word, wcount in sorted(counts.items(),
                                   key=lambda x: (x[1], x[0]), reverse=True):
            print('{}\t{}'.format(word, wcount), file=output_stream)
    logger.info('Done counting words in {}'.format(corpus_filepath))
    return counts


def _count(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.corpus)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    count(output_dirpath, args.corpus)


def main():
    """Launch entropix."""
    parser = argparse.ArgumentParser(prog='entropix')
    subparsers = parser.add_subparsers()
    parser_count = subparsers.add_parser(
        'count', formatter_class=argparse.RawTextHelpFormatter,
        help='count words in input corpus')
    parser_count.add_argument('-c', '--corpus', required=True,
                              help='an input .txt corpus to compute counts on')
    parser_count.add_argument('-o', '--output',
                              help='absolute path to output directory. '
                                   'If not set, will default to corpus dir')
    parser_count.set_defaults(func=_count)
    parser_compute = subparsers.add_parser(
        'compute', formatter_class=argparse.RawTextHelpFormatter,
        help='compute entropy from input .vocab file')
    parser_compute.set_defaults(func=_compute)
    parser_compute.add_argument('-c', '--counts', required=True,
                                help='input .counts counts file to compute '
                                     'entropy from')
    args = parser.parse_args()
    args.func(args)
