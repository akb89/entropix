"""Welcome to entropix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import entropix.utils.config as cutils
import entropix.core.counter as counter
import entropix.core.calculator as calculator
import entropix.core.evaluator as evaluator
import entropix.core.generator as generator
import entropix.core.reducer as reducer

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _evaluate(args):
    logger.info('Evaluating model against MEN {}'.format(args.model))
    evaluator.evaluate_distributional_space(args.model, args.vocab)


def _compute_entropy(args):
    logger.info('Computing entropy from file {}'.format(args.counts))
    counts = {}
    with open(args.counts, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            word_count = line.split('\t')
            counts[word_count[0]] = int(word_count[1])
    calculator.compute_entropy(counts)


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
        counts = counter.count_words(corpus_filepath=args.corpus,
                                     min_count=args.min_count,
                                     output_dirpath=output_dirpath)
    else:
        counts = counter.count_words(corpus_filepath=args.corpus,
                                     min_count=args.min_count)
    logger.info('Corpus size = {}'.format(sum(counts.values())))
    logger.info('Vocab size = {}'.format(len(counts)))


def _generate(args):
    logger.info('Generating distributional model from {}'.format(args.corpus))
    if not args.output:
        output_dirpath = os.path.dirname(args.corpus)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    generator.generate_distributional_model(output_dirpath, args.corpus,
                                            args.min_count, args.win_size)


def _reduce(args):
    logger.info('Applying SVD to model {}'.format(args.model))
    model_basename = args.model.split('.npz')[0]
    dense_model_filepath = '{}.dense.npz'.format(model_basename)
    diag_matrix_filepath = '{}.diag.npz'.format(model_basename)
    reducer.reduce_matrix_via_svd(args.model, args.dim, dense_model_filepath,
                                  diag_matrix_filepath)


def _compute_pairwise_cosines(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.model)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    calculator.compute_pairwise_cosine_sim(output_dirpath, args.model,
                                           args.num_threads, args.bin_size,
                                           args.vocab)


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
        help='compute entropy or pairwise cosine similarity metrics')
    compute_sub = parser_compute.add_subparsers()
    parser_compute_entropy = compute_sub.add_parser(
        'entropy', formatter_class=argparse.RawTextHelpFormatter,
        help='compute entropy from input .counts file')
    parser_compute_entropy.set_defaults(func=_compute_entropy)
    parser_compute_entropy.add_argument(
        '-c', '--counts', required=True,
        help='input .counts counts file to compute entropy from')
    parser_compute_cosine = compute_sub.add_parser(
        'cosine', formatter_class=argparse.RawTextHelpFormatter,
        help='compute pairwise cosine similarity between vocabulary items')
    parser_compute_cosine.set_defaults(func=_compute_pairwise_cosines)
    parser_compute_cosine.add_argument(
        '-v', '--vocab', help='set of vocabulary items to compute the '
                              'distribution for. If not set, will default to '
                              'whole vocabulary')
    parser_compute_cosine.add_argument('-m', '--model', required=True,
                                       help='distributional space')
    parser_compute_cosine.add_argument(
        '-o', '--output', help='absolute path to output directory. If not '
                               'set, will default to space dir')
    parser_compute_cosine.add_argument('-n', '--num-threads', default=1,
                                       type=int, help='number of threads')
    parser_compute_cosine.add_argument('-b', '--bin-size', default=0.1,
                                       type=float, help='bin size for the '
                                                        'distribution output')
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
    parser_reduce = subparsers.add_parser(
        'reduce', formatter_class=argparse.RawTextHelpFormatter,
        help='apply svd to input matrix')
    parser_reduce.set_defaults(func=_reduce)
    parser_reduce.add_argument('-m', '--model', required=True,
                               help='absolute path to .npz matrix '
                                    'corresponding to the distributional '
                                    'space to reduce')
    parser_reduce.add_argument('-k', '--dim', default=0, type=int,
                               help='number of dimensions in final model')
    args = parser.parse_args()
    args.func(args)
