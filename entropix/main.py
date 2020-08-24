"""Welcome to entropix.

This is the entry point of the application.
"""
import os

import argparse
import logging
import logging.config

import embeddix

import entropix.utils.config as cutils
import entropix.utils.data as dutils
import entropix.core.sampler as sampler

logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


# pylint: disable=R0914
def _sample(model, splits_dict, dataset, kfold_size, mode, metric, shuffle,
            max_num_threads, limit):
    if mode not in ['seq', 'limit']:
        raise Exception('Invalid sampling mode: {}'.format(mode))

    if mode == 'seq':
        logger.info('Sampling dimensions in seq mode over {} dims, '
                    'optimizing on {} using {}'
                    .format(model.shape[1], dataset, metric))
        sampled_dims = sampler.sample_seq(model, splits_dict, kfold_size,
                                          metric, shuffle, max_num_threads)
    if mode == 'limit':
        if kfold_size > 0:
            raise Exception('kfolding currently unsupported in limit mode')
        sampled_dims = sampler.sample_limit(model, splits_dict[1]['train'],
                                            metric, limit)
    results = {}
    for fold, dims in sampled_dims.items():
        results[fold] = {
            'dims': dims,
            'train': splits_dict[fold]['train'],
            'test': splits_dict[fold]['test']
        }
    return results


def sample(args):
    """Sample dimensions from model."""
    model = embeddix.load_dense(args.model)
    vocab = embeddix.load_vocab(args.vocab)
    splits_dict = dutils.load_splits_dict(args.dataset, vocab, args.kfold_size)
    results = _sample(model, splits_dict, args.dataset, args.kfold_size,
                      args.mode, args.metric, args.shuffle, args.num_threads,
                      args.limit)
    if args.dump:
        if args.kfold_size != 0:
            raise Exception('Cannot dump dims for kfolds')
        output_fpath = '{}.sampled.mode-{}.dataset-{}.metric-{}.dims'.format(
            os.path.abspath(args.model).split('.npy')[0], args.mode,
            args.dataset, args.metric)
        with open(output_fpath, 'w', encoding='utf-8') as output_str:
            for dim in sorted(results[1]['dims']):
                print(dim, file=output_str)


# pylint: disable=C0103
def restricted_kfold_size(x):
    """Restrict kfold-size values."""
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
        help='find best dim config that maximize dataset score')
    parser_sample.set_defaults(func=sample)
    parser_sample.add_argument('-m', '--model', required=True,
                               help='absolute path to .singvectors.npy file')
    parser_sample.add_argument('-v', '--vocab', required=True,
                               help='asbolute path to .vocab file')
    parser_sample.add_argument('-d', '--dataset', required=True,
                               choices=['men', 'simlex', 'simverb'],
                               help='dataset to optimize on')
    parser_sample.add_argument('-s', '--shuffle', action='store_true',
                               help='whether or not to shuffle at each iter')
    parser_sample.add_argument('-z', '--mode', choices=['seq', 'limit'],
                               help='which version of the algorithm to use')
    parser_sample.add_argument('-l', '--limit', type=int, default=0,
                               help='max number of dim in limit mode')
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
