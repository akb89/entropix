"""Evaluate a given Numpy distributional model against the MEN dataset."""
import logging
import numpy as np

import entropix.utils.metrix as metrix

__all__ = ('evaluate_distributional_space', 'evaluate', 'is_improving',
           'is_degrading')

logger = logging.getLogger(__name__)


def is_degrading(metric_value, best_metric_value, metric):
    if metric not in ['spr', 'rmse', 'combined']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric in ['spr', 'combined']:
        return metric_value < best_metric_value
    return metric_value > best_metric_value


def is_improving(metric_value, best_metric_value, metric):
    if metric not in ['spr', 'rmse', 'combined']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric in ['spr', 'combined']:
        return metric_value > best_metric_value
    return metric_value < best_metric_value  # for rmse we want to lower the loss


def evaluate(model, splits, dataset, metric, alpha=None):
    """Evaluate a given model against splits given a metric (spr or rmse)."""
    if metric not in ['spr', 'rmse', 'combined']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if metric == 'spr':
        return metrix.get_spr_correlation(
            model, splits['left_idx'], splits['right_idx'], splits['sim'],
            dataset)
    if metric == 'rmse':
        return metrix.get_rmse(model, splits['left_idx'], splits['right_idx'],
                               splits['sim'], dataset)
    return metrix.get_combined_spr_rmse(
        model, splits['left_idx'], splits['right_idx'], splits['sim'], dataset,
        alpha)


def evaluate_distributional_space(model, dataset, metric, model_type,
                                  vocab_filepath):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Evaluating distributional space...')
    if model_type == 'numpy':
        left_idx, right_idx, sim = load_words_and_sim(
            vocab_filepath, dataset, shuffle=False)
        splits = {
            'left_idx': left_idx,
            'right_idx': right_idx,
            'sim': sim
        }
        eval_metric = get_eval_metric(model, splits, dataset, metric,
                                      alpha=None)
    elif model_type == 'gensim':
        logger.info('Loading gensim vocabulary')
        left, right, sim = _load_left_right_sim(dataset)
        _sim = []
        for x, y, z in zip(left, right, sim):
            if x not in model.wv.vocab or y not in model.wv.vocab:
                logger.error('Could not find one of more pair item in model '
                             'vocabulary: {}, {}'.format(x, y))
                continue
            _sim.append(z)
        model_sim = _get_gensim_model_sim(model, left, right)
        if metric == 'rmse':
            if dataset == 'men':
                _sim = [x/50 for x in _sim]  # men has sim in [0, 50]
            else:
                _sim = [x/10 for x in _sim]  # all other datasets have sim in [0, 10]
            # for x, y in zip(_sim, model_sim):
            #     print(x, y)
            eval_metric = _rmse(np.array(_sim), np.array(model_sim))
        elif metric == 'spr':
            eval_metric = _spearman(_sim, model_sim)
    logger.info('{} = {}'.format(metric, eval_metric))
    return eval_metric
