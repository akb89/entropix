"""Welcome to entropix.

This is the entry point of the application.
"""
import os

import argparse
import datetime
import logging
import logging.config

import numpy as np

import entropix.utils.config as cutils
import entropix.utils.files as futils
import entropix.utils.data as dutils
import entropix.core.calculator as calculator
import entropix.core.comparator as comparator
import entropix.core.evaluator as evaluator
import entropix.core.generator as generator
import entropix.core.reducer as reducer
import entropix.core.remover as remover
import entropix.core.weigher as weigher
import entropix.core.overlap_analyser as analyser

from entropix.core.sampler import Sampler

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _evaluate(args):
    logger.info('Evaluating model on {}'.format(args.dataset))
    logger.info('Loading distributional space from {}'.format(args.model))
    model, vocab = dutils.load_model_and_vocab(
        args.model, args.type, args.vocab, args.singvalues, args.singalpha,
        args.start, args.end, args.dims)
    evaluator.evaluate_distributional_space(model, vocab, args.dataset,
                                            args.metric, args.type,
                                            args.distance, args.kfold_size)


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
                                            args.min_count, args.win_size,
                                            args.with_info, args.info_model,
                                            args.num_threads)


def _svd(args):
    logger.info('Applying SVD to model {}'.format(args.model))
    sing_values_filepath = futils.get_singvalues_filepath(args.model,
                                                          args.dim,
                                                          args.which,
                                                          args.dataset,
                                                          args.compact)
    sing_vectors_filepaths = futils.get_singvectors_filepath(args.model,
                                                             args.dim,
                                                             args.which,
                                                             args.dataset,
                                                             args.compact)
    reducer.apply_svd(args.model, args.dim, sing_values_filepath,
                      sing_vectors_filepaths, args.which,
                      dataset=args.dataset, vocab_filepath=args.vocab,
                      compact=args.compact)


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
                                           args.vocab, args.num_threads,
                                           args.bin_size)


def _weigh(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.model)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    weigher.weigh(output_dirpath, args.model, args.weighing_func)


def _compute_singvectors_distribution(args):
    if not args.output:
        output_dirpath = os.path.dirname(args.model)
    else:
        output_dirpath = args.output
    if not os.path.exists(output_dirpath):
        logger.info('Creating directory {}'.format(output_dirpath))
        os.makedirs(output_dirpath)
    else:
        logger.info('Saving to directory {}'.format(output_dirpath))
    calculator.compute_singvectors_distribution(output_dirpath, args.model, args.save)


def _reduce(args):
    reducer.reduce(args.sparse, args.dataset, args.vocab)


def _sample(args):
    if args.output:
        dirname = args.output
        os.makedirs(dirname, exist_ok=True)
    else:
        dirname = os.path.dirname(args.model)
    basename = os.path.basename(args.model).split('.singvectors.npy')[0]
    if args.mode == 'mix':
        keep_filepath_basename = os.path.join(
            dirname,
            '{}.{}.sampledims.metric-{}.mode-{}.rate-{}.niter-{}.start-{}.end-{}'
            .format(basename, args.dataset, args.metric, args.args.mode,
                    args.rate, args.iter, args.start, args.end))
    elif args.mode == 'seq':
        keep_filepath_basename = os.path.join(
            dirname,
            '{}.{}.sampledims.metric-{}.mode-{}.niter-{}.start-{}.end-{}'
            .format(basename, args.dataset, args.metric, args.mode, args.iter,
                    args.start, args.end))
    elif args.mode == 'limit':
        keep_filepath_basename = os.path.join(
            dirname,
            '{}.{}.sampledims.metric-{}.mode-{}.d-{}.start-{}.end-{}'.format(
                basename, args.dataset, args.metric, args.mode, args.limit,
                args.start, args.end))
    if args.shuffle:
        keep_filepath_basename = '{}.shuffled.timestamp-{}'.format(
            keep_filepath_basename, datetime.datetime.now().timestamp())
    if args.kfolding:
        if not args.shuffle:
            keep_filepath_basename = '{}.timestamp-{}'.format(
                keep_filepath_basename, datetime.datetime.now().timestamp())
    logger.info('Output basename = {}'.format(keep_filepath_basename))
    if args.debug:
        logger.debug('Debug mode activated')
    if args.logs_dirpath:
        os.makedirs(args.logs_dirpath, exist_ok=True)
    logger.debug('Outputing logs to {}'.format(args.logs_dirpath))
    sampler = Sampler(args.model, args.type, args.vocab, args.dataset,
                      keep_filepath_basename, args.iter, args.shuffle,
                      args.mode, args.rate, args.start, args.end,
                      args.reduce, args.limit, args.rewind,
                      args.kfolding, args.kfold_size, args.num_threads,
                      args.dev_type, args.debug, args.metric, args.alpha,
                      args.logs_dirpath, args.distance, args.singvalues,
                      args.singalpha)
    sampler.sample_dimensions()

def restricted_kfold_size(x):
    x = float(x)
    if x < 0.0 or x > 0.5:
        raise argparse.ArgumentTypeError('{} kfold-size not in range [0, 0.5]'
                                         .format(x))
    return x


def restricted_energy(x):
    x = float(x)
    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError('{} energy not in range [0, 100]'
                                         .format(x))
    return x


def restricted_alpha(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError('{} alpha not in range [0.0, 1.0]'
                                         .format(x))
    return x


def _remove_mean(args):
    logger.info('Removing mean vector from {}'.format(args.model))
    output_filepath = '{}.nomean.npy'.format(args.model.split('.npy')[0])
    model = np.load(args.model)
    remover.remove_mean(model, output_filepath)


def _ica(args):
    reducer.apply_fast_ica(args.model, args.dataset, args.vocab, args.max_iter)


def _nmf(args):
    reducer.apply_nmf(args.model, args.init, args.max_iter, args.shuffle,
                      args.n_components, args.dataset, args.vocab)


def _analyse_ppmi_rows_overlap(args):
    analyser.analyse_overlap(args.model, args.vocab, args.dataset)


def _export(args):
    logger.info('Exporting file {} to {}.npy'
                .format(args.model, args.output))
    if args.type == 'numpy' and not args.vocab:
        raise Exception('--vocab is required with --format numpy')
    if args.dims and args.start or args.dims and args.end:
        raise Exception('Cannot specify both --dims and --start or --end params')
    if args.start is not None and args.end is None or args.end is not None and args.start is None:
        raise Exception('Both --start and --end params should be specified')
    model, vocab = dutils.load_model_and_vocab(
        args.model, args.type, args.vocab, args.singvalues, args.singalpha,
        args.start, args.end, args.dims, args.shuffle, args.randomize)
    np.save(args.output, model)
    dutils.save_vocab(vocab, '{}.vocab'.format(args.output))


def _compare(args):
    basename1 = os.path.basename(args.model1)
    basename2 = os.path.basename(args.model2)
    logger.info('Comparing DS models {} and {}'.format(basename1, basename2))
    VOCAB1_FILEPATH = '{}.vocab'.format(args.model1.split('.npy')[0])
    VOCAB2_FILEPATH = '{}.vocab'.format(args.model2.split('.npy')[0])
    model1, vocab1 = dutils.load_model_and_vocab(
        model_filepath=args.model1, model_type='numpy',
        vocab_filepath=VOCAB1_FILEPATH)
    model2, vocab2 = dutils.load_model_and_vocab(
        model_filepath=args.model2, model_type='numpy',
        vocab_filepath=VOCAB2_FILEPATH)
    avg, std = comparator.compare(model1, model2, vocab1, vocab2,
                                  args.num_neighbors, args.num_threads)
    logger.info('avg = {}'.format(avg))
    logger.info('std = {}'.format(std))


def main():
    """Launch entropix."""
    parser = argparse.ArgumentParser(prog='entropix')
    subparsers = parser.add_subparsers()
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
    parser_evaluate.add_argument('-d', '--dataset', required=True,
                                 choices=['ap', 'battig', 'essli', 'men',
                                          'simlex', 'simverb', 'sts2012',
                                          'ws353'],
                                 help='which dataset to evaluate on')
    parser_evaluate.add_argument('-i', '--dims',
                                 help='absolute path to .txt file containing'
                                      'a shortlist of dimensions, one per line'
                                      'to select from')
    parser_evaluate.add_argument('-s', '--start', type=int,
                                 help='index of singvectors dim to start from')
    parser_evaluate.add_argument('-e', '--end', type=int,
                                 help='index of singvectors dim to end at')
    parser_evaluate.add_argument('-t', '--type', choices=['numpy', 'gensim',
                                                          'ica', 'nmf', 'txt',
                                                          'scipy'],
                                 help='model type')
    parser_evaluate.add_argument('-c', '--metric',
                                 choices=['spr', 'rmse'],
                                 help='which eval metric to use')
    parser_evaluate.add_argument('-a', '--distance',
                                 choices=['cosine', 'euclidean'],
                                 help='which distance to use for similarity')
    parser_evaluate.add_argument('-x', '--kfold-size',
                                 type=restricted_kfold_size, default=0,
                                 help='determine size of kfold. Should be in '
                                      '[0, 0.5], that is, less than 50% of '
                                      'total dataset size')
    parser_evaluate.add_argument('--singvalues', default=None,
                                 help='absolute path to singular values')
    parser_evaluate.add_argument('--singalpha', type=float, default=0,
                                 help='power alpha for singular values')
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
    parser_generate.add_argument('-i', '--with-info', action='store_true',
                                 help='Whether or not to use informativeness')
    parser_generate.add_argument('-f', '--info-model',
                                 help='Absolute path to gensim cbow model')
    parser_generate.add_argument('-n', '--num-threads', type=int, default=1,
                                 help='number of threads to use for parallel '
                                      'processing with informativeness only')
    parser_reduce = subparsers.add_parser(
        'reduce', formatter_class=argparse.RawTextHelpFormatter,
        help='reduce a sparse matrix to a dense one containing the rows of '
             'the dataset only')
    parser_reduce.set_defaults(func=_reduce)
    parser_reduce.add_argument('-m', '--sparse', required=True,
                               help='absolute path to .npz sparse matrix')
    parser_reduce.add_argument('-d', '--dataset', required=True,
                               choices=['men', 'simlex', 'simverb'],
                               help='name of dataset')
    parser_reduce.add_argument('-v', '--vocab', required=True,
                               help='absolute path to vocabulary')
    parser_svd = subparsers.add_parser(
        'svd', formatter_class=argparse.RawTextHelpFormatter,
        help='apply svd to input matrix')
    parser_svd.set_defaults(func=_svd)
    parser_svd.add_argument('-m', '--model', required=True,
                            help='absolute path to .npz matrix '
                                 'corresponding to the distributional '
                                 'space to reduce via svd')
    parser_svd.add_argument('-k', '--dim', default=0, type=int,
                            help='number of dimensions in final model')
    parser_svd.add_argument('-w', '--which', choices=['LM', 'SM'],
                            default='LM',
                            help='Which k singular values to find:'
                                 'LM : largest singular values'
                                 'SM : smallest singular values')
    parser_svd.add_argument('-d', '--dataset',
                            choices=['men', 'simlex', 'simverb'],
                            help='if specified, will perform SVD only on '
                                 'the words contained in the dataset')
    parser_svd.add_argument('-v', '--vocab',
                            help='absolute path to vocabulary. '
                                 'To be specified only with --dataset')
    parser_svd.add_argument('-c', '--compact', action='store_true',
                            help='whether or not to store a compact matrix')
    parser_ica = subparsers.add_parser(
        'ica', formatter_class=argparse.RawTextHelpFormatter,
        help='apply FastICA to input sparse matrix')
    parser_ica.set_defaults(func=_ica)
    parser_ica.add_argument('-m', '--model', required=True,
                            help='absolute path to .npz matrix '
                                 'corresponding to the ppmi '
                                 'count matrix to apply ica to')
    parser_ica.add_argument('-d', '--dataset', required=True,
                            choices=['men', 'simlex', 'simverb'],
                            help='perform ICA only on '
                                 'the words contained in the dataset')
    parser_ica.add_argument('-v', '--vocab', required=True,
                            help='absolute path to vocabulary')
    parser_ica.add_argument('-a', '--max-iter', type=int, default=1000,
                            help='maximum number of iterations before timing out')
    parser_nmf = subparsers.add_parser(
        'nmf', formatter_class=argparse.RawTextHelpFormatter,
        help='apply NMF to input sparse matrix')
    parser_nmf.set_defaults(func=_nmf)
    parser_nmf.add_argument('-m', '--model', required=True,
                            help='absolute path to .npz matrix '
                                 'corresponding to the ppmi '
                                 'count matrix to apply ica to')
    parser_nmf.add_argument('-d', '--dataset',
                            choices=['men', 'simlex', 'simverb'],
                            help='perform NMF only on '
                                 'the words contained in the dataset')
    parser_nmf.add_argument('-v', '--vocab',
                            help='absolute path to vocabulary')
    parser_nmf.add_argument('-n', '--n-components', type=int,
                            help='number of components. If not set all features are kept')
    parser_nmf.add_argument('-i', '--init', default='random',
                            choices=['random', 'nndsvd', 'nndsvda', 'nndsvdar'],
                            help='method used to initialize the procedure')
    parser_nmf.add_argument('-a', '--max-iter', type=int, default=1000,
                            help='maximum number of iterations before timing out')
    parser_nmf.add_argument('--shuffle', action='store_true',
                            help='randomize the order of coordinates in the CD solver')
    parser_weigh = subparsers.add_parser(
        'weigh', formatter_class=argparse.RawTextHelpFormatter,
        help='weigh sparse matrix according to weighing function')
    parser_weigh.set_defaults(func=_weigh)
    parser_weigh.add_argument('-m', '--model', required=True,
                              help='absolute path to .npz matrix '
                              'corresponding to the distributional '
                              'space to weigh')
    parser_weigh.add_argument('-o', '--output',
                              help='absolute path to output directory. '
                              'If not set, will default to model dir')
    parser_weigh.add_argument('-w', '--weighing-func', choices=['ppmi'],
                              help='weighing function')
    parser_remove = subparsers.add_parser(
        'remove', formatter_class=argparse.RawTextHelpFormatter,
        help='remove the mean vector')
    parser_remove.set_defaults(func=_remove_mean)
    parser_remove.add_argument('-m', '--model', required=True,
                               help='absolute path to .npy matrix')
    parser_sample = subparsers.add_parser(
        'sample', formatter_class=argparse.RawTextHelpFormatter,
        help='find min num of dimensions that maximize dataset score')
    parser_sample.set_defaults(func=_sample)
    parser_sample.add_argument('-m', '--model', required=True,
                               help='absolute path to .singvectors.npy')
    parser_sample.add_argument('-v', '--vocab', required=True,
                               help='vocabulary mapping for dsm')
    parser_sample.add_argument('-o', '--output',
                               help='absolute path to output directory where '
                                    'to save sampled models')
    parser_sample.add_argument('-d', '--dataset', required=True,
                               choices=['men', 'simlex', 'simverb', 'sts2012'],
                               help='dataset to optimize on')
    parser_sample.add_argument('-i', '--iter', type=int, default=1,
                               help='number of iterations')
    parser_sample.add_argument('-s', '--shuffle', action='store_true',
                               help='whether or not to shuffle at each iter')
    parser_sample.add_argument('-z', '--mode', choices=['seq', 'mix', 'limit'],
                               default='seq',
                               help='which version of the algorithm to use')
    parser_sample.add_argument('-t', '--rate', type=int, default=100,
                               help='reduce every r dim in mix mode')
    parser_sample.add_argument('-b', '--start', type=int, default=0,
                               help='index of singvectors dim to start from')
    parser_sample.add_argument('-e', '--end', type=int, default=0,
                               help='index of singvectors dim to end at')
    parser_sample.add_argument('-r', '--reduce', action='store_true',
                               help='if set, will apply reduce_dim in seq mode')
    parser_sample.add_argument('-l', '--limit', type=int, default=5,
                               help='max number of dim in limit mode')
    parser_sample.add_argument('-w', '--rewind', action='store_true',
                               help='if set, will rewind in limit mode')
    parser_sample.add_argument('-k', '--kfolding', action='store_true',
                               help='if set, will sample with kfold')
    parser_sample.add_argument('-x', '--kfold-size',
                               type=restricted_kfold_size, default=0,
                               help='determine size of kfold. Should be in '
                                    '[0, 0.5], that is, less than 50% of '
                                    'total dataset size')
    parser_sample.add_argument('-y', '--dev-type', default='nodev',
                               choices=['nodev', 'regular', 'balanced'],
                               help='which type of dev split to use')
    parser_sample.add_argument('-c', '--metric', required=True,
                               choices=['spr', 'rmse', 'combined', 'both'],
                               help='which eval metric to use')
    parser_sample.add_argument('-a', '--alpha', type=restricted_alpha,
                               help='how to weight combined spr and rmse eval '
                                    'metrics. alpha < 0.5 will bias eval '
                                    'towards rmse. alpha > 0.5 will bias eval'
                                    'towards spr')
    parser_sample.add_argument('-n', '--num-threads', type=int, default=1,
                               help='number of threads to use for parallel '
                                    'processing of kfold validation')
    parser_sample.add_argument('--debug', action='store_true',
                               help='activate debugger')
    parser_sample.add_argument('--logs-dir', dest='logs_dirpath',
                               help='absolute path to logs directory')
    parser_sample.add_argument('--distance', required=True,
                               choices=['cosine', 'euclidean'],
                               help='which distance to use for similarity')
    parser_sample.add_argument('--singvalues',
                               help='absolute path to singular values')
    parser_sample.add_argument('--singalpha', type=float,
                               default=0,
                               help='power alpha for singular values')
    parser_sample.add_argument('--type', required=True,
                               choices=['numpy', 'gensim', 'ica', 'nmf', 'txt',
                                        'scipy'],
                               help='model type')
    parser_analyse_ppmi_rows_overlap = subparsers.add_parser(
        'analyse-overlap', formatter_class=argparse.RawTextHelpFormatter,
        help='provides qualitative data on features overlap in a provided dataset')
    parser_analyse_ppmi_rows_overlap.set_defaults(func=_analyse_ppmi_rows_overlap)
    parser_analyse_ppmi_rows_overlap.add_argument('-m', '--model', required=True,
                                                  help='absolute path to .npz matrix')
    parser_analyse_ppmi_rows_overlap.add_argument('-v', '--vocab', required=True,
                                                  help='vocabulary mapping for dsm')
    parser_analyse_ppmi_rows_overlap.add_argument('-d', '--dataset', required=True,
                                                  choices=['men', 'simlex', 'simverb'],
                                                  help='which dataset to consider')
    parser_export = subparsers.add_parser(
        'export', formatter_class=argparse.RawTextHelpFormatter,
        help='export different model types to a standardized numpy format')
    parser_export.set_defaults(func=_export)
    parser_export.add_argument('-m', '--model', required=True,
                               help='absolute path to input embedding model')
    parser_export.add_argument('-o', '--output', required=True,
                               help='absolute path to output numpy model')
    parser_export.add_argument('-t', '--type', required=True,
                               choices=['numpy', 'gensim', 'txt'],
                               help='input embedding format type')
    parser_export.add_argument('-v', '--vocab',
                               help='vocabulary mapping for dsm')
    parser_export.add_argument('-d', '--dims',
                               help='absolute path to .txt file containing'
                                    'a shortlist of dimensions, one per line'
                                    'to select from')
    parser_export.add_argument('-s', '--start', type=int,
                               help='index of singvectors dim to start from')
    parser_export.add_argument('-e', '--end', type=int,
                               help='index of singvectors dim to end at')
    parser_export.add_argument('--singvalues',
                               help='absolute path to singular values')
    parser_export.add_argument('--singalpha', type=float,
                               help='power alpha for singular values')
    parser_export.add_argument('--shuffle', action='store_true',
                               help='if true, will shuffle svd dims')
    parser_export.add_argument('--randomize', action='store_true',
                               help='if true, will replace vectors values '
                                    'with random numbers')
    parser_compare = subparsers.add_parser(
        'compare', formatter_class=argparse.RawTextHelpFormatter,
        help='compare the nearest neighbors of two numpy models')
    parser_compare.set_defaults(func=_compare)
    parser_compare.add_argument('-m1', '--model1', required=True,
                                help='absolute path to input embedding model1')
    parser_compare.add_argument('-m2', '--model2', required=True,
                                help='absolute path to input embedding model2')
    parser_compare.add_argument('-n', '--num-neighbors', type=int, default=5,
                                help='number of neighbors to consider')
    parser_compare.add_argument('--num-threads', type=int, default=1,
                                help='number of threads to use for low RAM')
    args = parser.parse_args()
    args.func(args)
