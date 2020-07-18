"""Dimensionality reduction through dimensionality selection."""
import logging
import random
import functools
import multiprocessing

import numpy as np

import entropix.core.evaluator as evaluator

__all__ = ('sample_seq', 'sample_limit')

logger = logging.getLogger(__name__)


def _init_eval_metric(metric):
    if metric not in ['spr', 'rmse', 'pearson', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric in ['spr', 'pearson']:
        return 0
    if metric == 'rmse':
        return 10**15.
    return (0, 10**15)


# pylint: disable=C0103,W0621
def sample_limit(model, train_splits, metric, limit):
    """Sample dimensions in limit mode."""
    best_metric = _init_eval_metric(metric)
    dims = []
    max_num_dim_best = 0
    alldims = list(range(model.shape[1]))
    for k in range(limit):
        best_dim_idx = -1
        least_worst_dim = -1
        least_worst_metric = _init_eval_metric(metric)
        for dim_idx in alldims:
            if dim_idx in dims:
                continue
            dims.append(dim_idx)
            eval_metric = evaluator.evaluate(
                model[:, dims], train_splits, metric=metric)
            if evaluator.is_improving(eval_metric,
                                      best_metric,
                                      metric=metric):
                best_metric = eval_metric
                best_dim_idx = dim_idx
            elif evaluator.is_improving(eval_metric,
                                        least_worst_metric,
                                        metric=metric):
                least_worst_metric = eval_metric
                least_worst_dim = dim_idx
            dims.pop()
        if best_dim_idx == -1:
            logger.info('Could not find a better metric with {} '
                        'dims. Added least worst dim to continue'.format(k+1))
            dims.append(least_worst_dim)
        else:
            dims.append(best_dim_idx)
            max_num_dim_best = len(dims)
            logger.info('Current best {} = {} with dims = {}'
                        .format(metric, best_metric, dims))
    logger.info('Best eval metrix = {} with ndims = {}'
                .format(best_metric, max_num_dim_best))
    return {1: dims}  # to remain consistent with sample_seq return


def sample_seq_reduce(splits_dict, dims, step, fold, metric,
                      best_train_eval_metric):
    """Remove dimensions that do not negatively impact scores on train."""
    logger.info('Reducing dimensions while maintaining highest score '
                'on eval metric {}. Step = {}. Best train eval metric = {}'
                .format(metric, step, best_train_eval_metric))
    dims_set = set(dims)
    for dim_idx in dims:
        dims_set.remove(dim_idx)
        train_eval_metric = evaluator.evaluate(
            model[:, list(dims_set)], splits_dict[fold]['train'],
            metric=metric)
        if evaluator.is_degrading(train_eval_metric,
                                  best_train_eval_metric,
                                  metric=metric):
            dims_set.add(dim_idx)
            continue
        logger.info('Constant best train {} = {} for fold {} removing '
                    'dim_idx = {}. New ndim = {}'
                    .format(metric, train_eval_metric, fold,
                            dim_idx, len(dims_set)))
        best_train_eval_metric = train_eval_metric
    logger.info('Finished reducing dims')
    keep = list(sorted(dims_set, key=dims.index))
    if len(keep) != len(dims):
        step += 1
        keep, best_train_eval_metric = sample_seq_reduce(
            splits_dict, keep, step, fold, metric,
            best_train_eval_metric)
    return keep, best_train_eval_metric


def sample_seq_add(splits_dict, keep, alldims, metric, fold,
                   best_train_eval_metric):
    """Add dimensions that improve scores on train."""
    logger.info('Increasing dimensions to maximize score on eval metric '
                '{}. Best train eval metric = {}'
                .format(metric, best_train_eval_metric))
    dims = [idx for idx in alldims if idx not in keep]
    added_counter = 0
    for idx, dim_idx in enumerate(dims):
        keep.append(dim_idx)
        train_eval_metric = evaluator.evaluate(
            model[:, keep], splits_dict[fold]['train'], metric=metric)
        if evaluator.is_improving(train_eval_metric,
                                  best_train_eval_metric,
                                  metric=metric):
            added_counter += 1
            best_train_eval_metric = train_eval_metric
            logger.info('New best train {} = {} on fold {} with ndim = {} '
                        'at idx = {} and dim_idx = {}'.format(
                            metric, best_train_eval_metric, fold,
                            len(keep), idx, dim_idx))
        else:
            keep.pop()
    return keep, best_train_eval_metric


def _sample_seq(splits_dict, keep, alldims, metric, fold):
    best_train_eval_metric = evaluator.evaluate(
        model[:, keep], splits_dict[fold]['train'], metric=metric)
    logger.debug('Initial train eval metric = {}'.format(
        best_train_eval_metric))
    keep, best_train_eval_metric = sample_seq_add(
        splits_dict, keep, alldims, metric, fold,
        best_train_eval_metric)
    keep, best_train_eval_metric = sample_seq_reduce(
        splits_dict, keep, 1, fold, metric, best_train_eval_metric)
    return fold, keep


# pylint: disable=W0601
def sample_seq(_model, splits_dict, kfold_size, metric, shuffle,
               max_num_threads):
    """Sample dimensions in sequential mode."""
    global model  # ugly hack to reuse same in-memory model during forking
    model = _model
    alldims = list(range(model.shape[1]))
    if shuffle:
        random.shuffle(alldims)
        keep = np.random.choice(list(range(model.shape[1])),
                                size=2, replace=False).tolist()
    else:
        keep = [0, 1]  # select the first two
    # sample dimensons multi-threaded on all kfolds
    num_folds = len(splits_dict.keys())
    logger.info('Applying kfolding with k={} folds where '
                'each test fold is of size {} and accounts for '
                '{}% of the data'
                .format(num_folds, len(splits_dict[1]['test']['sim']),
                        kfold_size*100))
    num_threads = num_folds if num_folds <= max_num_threads \
        else max_num_threads
    with multiprocessing.Pool(num_threads) as pool:
        _sample = functools.partial(_sample_seq, splits_dict, keep,
                                    alldims, metric)
        sampled_dims = {}
        for fold, keep in pool.imap_unordered(_sample,
                                              range(1, num_folds+1)):
            sampled_dims[fold] = keep
    return sampled_dims
