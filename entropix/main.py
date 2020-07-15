"""Welcome to entropix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import entropix.utils.config as cutils

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _sample(model_filepath, vocab_filepath, dataset, kfold_size, mode):
    if mode not in ['seq', 'limit']:
        raise Exception('Invalid sampling mode: {}'.format(mode))
    model = embeddix.load_sparse(model_filepath)
    vocab = embeddix.load_vocab(vocab_filepath)
    splits_dict = dutils.load_splits(dataset, vocab, kfold_size)
    if mode == 'seq':
        logger.info('Sampling dimensions in seq mode over {} dims, '
                    'optimizing on {} using {}'.format())
        sampled_dims = sampler.sample_seq()
    if mode == 'limit':
        if kfolding:
            raise Exception('kfolding currently unsupported in limit mode')
        sampled_dims = sampler.sample_limit()
    results = {}
    for fold, dims in sampled_dims.items():
        results[fold]['dim'] = dims
        results[fold]['train'] = evaluator.evaluate(
            model[:, dims], splits_dict[fold]['train'], metric=metric)
        results[fold]['test'] = evaluator.evaluate(
            model[:, dims], splits_dict[fold]['test'], metric=metric)
    return results


def sample(args):
    """Sample dimensions from model."""
    pass


def restricted_kfold_size(x):
    x = float(x)
    if x < 0.0 or x > 0.5:
        raise argparse.ArgumentTypeError('{} kfold-size not in range [0, 0.5]'
                                         .format(x))
    return x


def main():
    """Launch entropix."""
    parser = argparse.ArgumentParser(prog='entropix')
    subparsers = parser.add_subparsers()
    parser_sample = subparsers.add_parser(
        'sample', formatter_class=argparse.RawTextHelpFormatter,
        help='find min num of dimensions that maximize dataset score')
    parser_sample.set_defaults(func=_sample)
    parser_sample.add_argument('-m', '--model', required=True,
                               help='absolute path to .singvectors.npy file')
    parser_sample.add_argument('-v', '--vocab', required=True,
                               help='asbolute path to .vocab file')
    parser_sample.add_argument('-d', '--datasets', required=True,
                               choices=['men', 'simlex', 'simverb'],
                               help='dataset to optimize on')
    parser_sample.add_argument('-s', '--shuffle', action='store_true',
                               help='whether or not to shuffle at each iter')
    parser_sample.add_argument('-z', '--mode', choices=['seq', 'limit'],
                               help='which version of the algorithm to use')
    parser_sample.add_argument('-l', '--limit', type=int, default=5,
                               help='max number of dim in limit mode')
    parser_sample.add_argument('-k', '--kfolding', action='store_true',
                               help='if set, will sample with kfold')
    parser_sample.add_argument('-x', '--kfold-size',
                               type=restricted_kfold_size, default=0,
                               help='determine size of kfold. Should be in '
                                    '[0, 0.5], that is, less than 50% of '
                                    'total dataset size')
    parser_sample.add_argument('-c', '--metric', required=True,
                               choices=['spr', 'rmse', 'pearson', 'both'],
                               help='which eval metric to use. '
                                    'both is spr+rmse')
    parser_sample.add_argument('-n', '--num-threads', type=int, default=1,
                               help='number of threads to use for parallel '
                                    'processing of kfold validation')
    parser_sample.add_argument('--dump', action='store_true',
                               help='if set, will save sampled dims to file')
    args = parser.parse_args()
    args.func(args)
