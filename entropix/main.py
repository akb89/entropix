"""Welcome to entropix.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

from collections import defaultdict

import entropix.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _compute_vocab(corpus, vocab_filepath):
    vocab = defaultdict(int)
    with open(corpus, 'r') as corpus_stream:
        for line in corpus_stream:
            tokens = line.strip().split()
            for token in tokens:
                vocab[token] += 1
    logger.info('Saving vocabulary to {}'.format(vocab_filepath))
    with open(vocab_filepath, 'w') as vocab_stream:
        for item in sorted(vocab.items(), key=lambda x: (x[1].count, x[0]),
                           reverse=True):
            print('{}\t{}'.format(item[0], item[1]), file=vocab_stream)


def _count(args):
    if not args.vocab:
        logger.info('No vocabulary file specified. Computing vocabulary with counts...')
        vocab_filepath = '{}.vocab'.format(args.model)
        _compute_vocab(args.corpus, vocab_filepath)


def main():
    """Launch entropix."""
    parser = argparse.ArgumentParser(prog='entropix')
    subparsers = parser.add_subparsers()
    parser_count = subparsers.add_parser(
        'count', formatter_class=argparse.RawTextHelpFormatter,
        help='count co-occurences in input corpus')
    parser_count.add_argument('--corpus',
                              help='an input corpus to compute counts from')
    parser_count.add_argument('--vocab',
                              help='corpus vocabulary with frequencies')
    parser_count.add_arguments('--model', help='absolute path to output model')
    parser_count.set_defaults(func=_count)
    args = parser.parse_args()
    args.func(args)
