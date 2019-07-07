"""Evaluate a given Numpy distributional model against the MEN dataset."""
import logging

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
    if metrix == 'combined':
        return metrix.get_combined_spr_rmse(
            model, splits['left_idx'], splits['right_idx'], splits['sim'],
            dataset, alpha, distance)
    return metrix.get_both_spr_rmse(
        model, splits['left_idx'], splits['right_idx'], splits['sim'],
        dataset, distance)


def evaluate_distributional_space(model, dataset, metric, model_type,
                                  vocab_filepath, distance):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Evaluating distributional space...')
    if model_type not in ['numpy', 'gensim', 'ica']:
        raise Exception('Unsupporteed model-type: {}'.format(model_type))
    if metric not in ['spr', 'rmse']:
        raise Exception('Unsupported metric: {}'.format(metric))
    if model_type in ['numpy', 'ica']:
        vocab = dutils.load_vocab(vocab_filepath)
        left_idx, right_idx, sim = dutils.load_dataset(dataset, vocab)
        if metric == 'rmse':
            eval_metric = metrix.get_rmse(model, left_idx, right_idx, sim,
                                          dataset, distance)
        elif metric == 'spr':
            eval_metric = metrix.get_spr_correlation(model, left_idx,
                                                     right_idx, sim,
                                                     dataset, distance)
    # elif model_type == 'gensim':
    #     logger.info('Loading gensim vocabulary')
    #     left, right, sim = _load_left_right_sim(dataset)
    #     _sim = []
    #     for x, y, z in zip(left, right, sim):
    #         if x not in model.wv.vocab or y not in model.wv.vocab:
    #             logger.error('Could not find one of more pair item in model '
    #                          'vocabulary: {}, {}'.format(x, y))
    #             continue
    #         _sim.append(z)
    #     model_sim = _get_gensim_model_sim(model, left, right)
    #     if metric == 'rmse':
    #         if dataset == 'men':  # men has sim in [0, 50]
    #             _sim = [x/50 for x in _sim]
    #         else:  # all other datasets have sim in [0, 10]
    #             _sim = [x/10 for x in _sim]
    #         # for x, y in zip(_sim, model_sim):
    #         #     print(x, y)
    #         eval_metric = _rmse(np.array(_sim), np.array(model_sim))
    #     elif metric == 'spr':
    #         eval_metric = _spearman(_sim, model_sim)
    # logger.info('{} = {}'.format(metric, eval_metric))
    return eval_metric
