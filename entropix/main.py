"""Welcome to entropix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import entropix.utils.config as cutils
import entropix.core.count as count
import entropix.core.compute as compute
import entropix.core.generate as generate
import entropix.core.cosine_distribution as cosine_distribution

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _compute(args):
    logger.info('Computing entropy from file {}'.format(args.counts))
    counts = {}
    with open(args.counts, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            word_count = line.split('\t')
            counts[word_count[0]] = int(word_count[1])
    compute.compute_entropy(counts)


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
    count.count_words(output_dirpath, args.corpus)


def _generate(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.corpus)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    generate.generate_model(output_dirpath, args.corpus, args.threshold)


def _compute_pairwise_cosines(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.space)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    if not args.vocabulary:
        vocabulary = ''
    else:
        vocabulary = args.vocabulary
    cosine_distribution.cosine_distribution(output_dirpath, args.space,
                                            args.threads_number, args.bin_size,
                                            vocabulary)


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
        help='compute entropy from input .counts file')
    parser_compute.set_defaults(func=_compute)
    parser_compute.add_argument('-c', '--counts', required=True,
                                help='input .counts counts file to compute '
                                     'entropy from')
    parser_generate = subparsers.add_parser(
        'generate', formatter_class=argparse.RawTextHelpFormatter,
        help='generate raw frequency count based model')
    parser_generate.set_defaults(func=_generate)
    parser_generate.add_argument('-c', '--corpus', required=True,
                                 help='an input .txt corpus to compute \
                                 counts on')
    parser_generate.add_argument('-o', '--output',
                                 help='absolute path to output directory. '
                                 'If not set, will default to corpus dir')
    parser_generate.add_argument('-t', '--threshold', default=0, type=int,
                                 help='frequency threshold on vocabulary')
    parser_generate.add_argument('-w', '--winsize', default=2, type=int,
                                 help='size of context window')
    parser_cosine = subparsers.add_parser(
        '_compute_pairwise_cosines', formatter_class=argparse.RawTextHelpFormatter,
        help='generate pairiwise cosine similarity between vocabulary items')
    parser_cosine.set_defaults(func=_compute_pairwise_cosines)
    parser_cosine.add_argument('-v', '--vocabulary',
                               help='set of vocabulary items to compute '
                               'the distribution for.'
                               'If not set, will default to whole vocabulary')
    parser_cosine.add_argument('-s', '--space', required=True,
                               help='distributional space')
    parser_cosine.add_argument('-o', '--output',
                               help='absolute path to output directory. '
                               'If not set, will default to space dir')
    parser_cosine.add_argument('-t', '--threads_number', default=1, type=int,
                               help='number of threads')
    parser_cosine.add_argument('-b', '--bin_size', default=0.1, type=float,
                               help='bin size for the distribution output')
    args = parser.parse_args()
    args.func(args)
