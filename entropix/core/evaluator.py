"""Evaluate a given Numpy distributional model against the MEN dataset."""
import logging

import embeddix

__all__ = ('evaluate', 'is_improving', 'is_degrading')

logger = logging.getLogger(__name__)


def is_degrading(metric_value, best_metric_value, metric):
    """Return true if metric value is degrading."""
    if metric not in ['spr', 'rmse', 'pearson', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'both':
        spr = metric_value[0]
        rmse = metric_value[1]
        best_spr = best_metric_value[0]
        best_rmse = best_metric_value[1]
        return spr < best_spr or rmse > best_rmse
    if metric in ['spr', 'pearson']:
        return metric_value < best_metric_value
    return metric_value > best_metric_value


def is_improving(metric_value, best_metric_value, metric):
    """Return true if metric value improving."""
    if metric not in ['spr', 'rmse', 'pearson', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'both':
        spr = metric_value[0]
        rmse = metric_value[1]
        best_spr = best_metric_value[0]
        best_rmse = best_metric_value[1]
        return spr > best_spr and rmse < best_rmse
    if metric in ['spr', 'pearson']:
        return metric_value > best_metric_value
    # for rmse we want to lower the loss
    return metric_value < best_metric_value


def evaluate(model, splits, metric):
    """Evaluate model on dataset splits with metric."""
    if metric not in ['spr', 'rmse', 'pearson', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    left_vectors = model[splits['left_idx']]
    right_vectors = model[splits['right_idx']]
    model_sim = embeddix.similarity(left_vectors, right_vectors)
    if metric == 'spr':
        return embeddix.spearman(splits['sim'], model_sim)
    if metric == 'pearson':
        return embeddix.pearson(splits['sim'], model_sim)
    if metric == 'rmse':
        return embeddix.rmse(splits['sim'], model_sim)
    return (embeddix.spearman(splits['sim'], model_sim),
            embeddix.rmse(splits['sim'], model_sim))
