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
import entropix.core.evaluate as evaluate
import entropix.core.generate as generate

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _evaluate(args):
    logger.info('Evaluating model against MEN {}'.format(args.model))
    evaluate.evaluate_distributional_space(args.model, args.vocab)


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
    if args.save:
        if not args.output:
            output_dirpath = os.path.dirname(args.corpus)
        else:
            output_dirpath = args.output
        if not os.path.exists(output_dirpath):
            logger.info('Creating directory {}'.format(output_dirpath))
            os.makedirs(output_dirpath)
        else:
            logger.info('Saving to directory {}'.format(output_dirpath))
        count.count_words(corpus_filepath=args.corpus,
                          min_count=args.min_count,
                          output_dirpath=output_dirpath)
    else:
        count.count_words(corpus_filepath=args.corpus,
                          min_count=args.min_count)


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
    generate.generate_distributional_model(output_dirpath, args.corpus,
                                           args.min_count, args.win_size)


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
    parser_count.add_argument('-m', '--min-count', default=0, type=int,
                              help='omit words below this count in output'
                                   'vocabulary')
    parser_count.add_argument('-s', '--save', action='store_true',
                              help='save counts to output')
    parser_count.set_defaults(func=_count)
    parser_compute = subparsers.add_parser(
        'compute', formatter_class=argparse.RawTextHelpFormatter,
        help='compute entropy from input .counts file')
    parser_compute.set_defaults(func=_compute)
    parser_compute.add_argument('-c', '--counts', required=True,
                                help='input .counts counts file to compute '
                                     'entropy from')
    parser_evaluate = subparsers.add_parser(
        'evaluate', formatter_class=argparse.RawTextHelpFormatter,
        help='evaluate a given distributional space against the MEN dataset')
    parser_evaluate.set_defaults(func=_evaluate)
    parser_evaluate.add_argument('-m', '--model', required=True,
                                 help='absolute path to .npz matrix '
                                      'corresponding to the distributional '
                                      'space to evaluate')
    parser_evaluate.add_argument('-v', '--vocab', required=True,
                                 help='absolute path to .map vocabulary file')
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
    parser_generate.add_argument('-m', '--min-count', default=0, type=int,
                                 help='frequency threshold on vocabulary')
    parser_generate.add_argument('-w', '--win-size', default=2, type=int,
                                 help='size of context window')
    args = parser.parse_args()
    args.func(args)
