"""Evaluate a given Numpy distributional model against the MEN dataset."""
import logging
import numpy as np

import entropix.utils.metrix as metrix
import entropix.utils.data as dutils

__all__ = ('evaluate_distributional_space', 'evaluate', 'is_improving',
           'is_degrading')

logger = logging.getLogger(__name__)


def is_degrading(metric_value, best_metric_value, metric):
    """Return true if metric value is degrading."""
    if metric not in ['spr', 'rmse', 'combined', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'both':
        spr = float(metric_value.split('#')[0])
        rmse = float(metric_value.split('#')[1])
        best_spr = float(best_metric_value.split('#')[0])
        best_rmse = float(best_metric_value.split('#')[1])
        return spr < best_spr or rmse > best_rmse
    if metric in ['spr', 'combined']:
        return metric_value < best_metric_value
    return metric_value > best_metric_value


def is_improving(metric_value, best_metric_value, metric):
    """Return true if metric value improving."""
    if metric not in ['spr', 'rmse', 'combined', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'both':
        spr = float(metric_value.split('#')[0])
        rmse = float(metric_value.split('#')[1])
        best_spr = float(best_metric_value.split('#')[0])
        best_rmse = float(best_metric_value.split('#')[1])
        return spr > best_spr and rmse < best_rmse
    if metric in ['spr', 'combined']:
        return metric_value > best_metric_value
    # for rmse we want to lower the loss
    return metric_value < best_metric_value


def evaluate(model, splits, dataset, metric, distance, alpha=None):
    """Evaluate a given model against splits given a metric (spr or rmse)."""
    if metric not in ['spr', 'rmse', 'combined', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'spr':
        return metrix.get_spr_correlation(
            model, splits['left_idx'], splits['right_idx'], splits['sim'],
            dataset, distance)
    if metric == 'rmse':
        return metrix.get_rmse(model, splits['left_idx'], splits['right_idx'],
                               splits['sim'], dataset, distance)
    if metric == 'combined':
        return metrix.get_combined_spr_rmse(
            model, splits['left_idx'], splits['right_idx'], splits['sim'],
            dataset, alpha, distance)
    return metrix.get_both_spr_rmse(
        model, splits['left_idx'], splits['right_idx'], splits['sim'],
        dataset, distance)


def evaluate_distributional_space(model, vocab, dataset, metric, model_type,
                                  distance, kfold_size):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Evaluating distributional space...')
    results = []
    if model_type not in ['scipy', 'numpy', 'ica', 'nmf', 'txt']:
        raise Exception('Unsupporteed model-type: {}'.format(model_type))
    if metric not in ['spr', 'rmse']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if model_type == 'scipy':
        model = model.todense()
    dim = model.shape[1]
    splits = dutils.load_kfold_splits(vocab, dataset, kfold_size,
                                      dev_type='nodev',
                                      output_logpath=None)
    for fold in splits.keys():
        logger.info('Evaluating on {} word pairs'
                    .format(len(splits[fold]['test']['sim'])))
        results.append(evaluate(model, splits[fold]['test'], dataset,
                                metric, distance))
    if kfold_size == 0:
        logger.info('{} = {}'.format(metric, results[0]))
    else:
        logger.info('avg test {} = {}'.format(metric, np.mean(results)))
        logger.info('std test {} = {}'.format(metric, np.std(results)))
    logger.info('dim = {}'.format(dim))
