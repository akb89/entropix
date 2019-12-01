"""Evaluate a given Numpy distributional model against the MEN dataset."""
import logging
import numpy as np

from sklearn.cluster import KMeans

import entropix.utils.metrix as metrix
import entropix.utils.data as dutils

__all__ = ('evaluate_distributional_space', 'evaluate', 'is_improving',
           'is_degrading', 'evaluate_single_dataset')

logger = logging.getLogger(__name__)


def _is_degrading_single(metric_value, best_metric_value, metric):
    if metric == 'both':
        spr = metric_value[0]
        rmse = metric_value[1]
        best_spr = best_metric_value[0]
        best_rmse = best_metric_value[1]
        return spr < best_spr or rmse > best_rmse
    if metric in ['spr', 'combined']:
        return metric_value < best_metric_value
    return metric_value > best_metric_value


def is_degrading(metric_values, best_metric_values, metric):
    """Return true if metric value is degrading."""
    if metric not in ['spr', 'rmse', 'combined', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    for metric_value, best_metric_value in zip(metric_values,
                                               best_metric_values):
        if _is_degrading_single(metric_value, best_metric_value, metric):
            return True
    return False


def _is_improving_single(metric_value, best_metric_value, metric):
    if metric == 'both':
        # spr = float(metric_value.split('#')[0])
        # rmse = float(metric_value.split('#')[1])
        # best_spr = float(best_metric_value.split('#')[0])
        # best_rmse = float(best_metric_value.split('#')[1])
        spr = metric_value[0]
        rmse = metric_value[1]
        best_spr = best_metric_value[0]
        best_rmse = best_metric_value[1]
        return spr > best_spr and rmse < best_rmse
    if metric in ['spr', 'combined']:
        # return float(metric_value) > float(best_metric_value)
        return metric_value > best_metric_value
    # for rmse we want to lower the loss
    # return float(metric_value) < float(best_metric_value)
    return metric_value < best_metric_value


def is_improving(metric_values, best_metric_values, metric):
    """Return true if metric value improving."""
    if metric not in ['spr', 'rmse', 'combined', 'both']:
        raise Exception('Unsupported metric: {}'.format(metric))
    for metric_value, best_metric_value in zip(metric_values,
                                               best_metric_values):
        if not _is_improving_single(metric_value, best_metric_value, metric):
            return False
    return True


def evaluate_single_dataset(model, splits, dataset, metric, distance,
                            alpha=None):
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


def evaluate(model, splits, datasets, metric, distance, alpha=None):
    """Evaluate a given model against splits given a metric (spr or rmse)."""
    return [evaluate_single_dataset(model, splits[dataset], dataset, metric,
                                    distance, alpha)
            for dataset in datasets]


def _evaluate_concept_categorization(model, vocab, dataset):
    if dataset not in ['ap', 'battig', 'essli']:
        raise Exception('Invalid concept categorization dataset: {}'
                        .format(dataset))
    logger.info('Evaluating concept categorization on {}'.format(dataset))
    categories_to_words = dutils.load_dataset(dataset, vocab)
    categories = categories_to_words.keys()
    centroids = np.empty(shape=(len(categories), model.shape[1]))
    for idx, (_, words) in enumerate(categories_to_words.items()):
        word_idxx = [vocab[word] for word in words]
        centroids[idx] = np.mean(model[word_idxx, :], axis=0)
    category_words = [word for words in categories_to_words.values() for word in words]
    category_words_idx = [vocab[word] for word in category_words]
    pred_clusters = KMeans(init=centroids,
                           n_clusters=len(categories)).fit_predict(model[category_words_idx, :])
    true_clusters = np.array([idx for idx, words in enumerate(categories_to_words.values()) for _ in words])
    purity = metrix.purity(true_clusters, pred_clusters)
    logger.info('Cluster purity = {}'.format(purity))
    return purity


def _evaluate_word_similarity(model, vocab, dataset, metric, model_type,
                              distance, kfold_size):
    results = []
    if dataset not in ['men', 'simlex', 'simverb']:
        raise Exception('Invalid similarity dataset: {}'.format(dataset))
    if model_type not in ['scipy', 'numpy', 'ica', 'nmf', 'txt']:
        raise Exception('Unsupported model-type: {}'.format(model_type))
    if metric not in ['spr', 'rmse']:
        raise Exception('Unsupported metric: {}'.format(metric))
    logger.info('Evaluating word similarity on {}'.format(dataset))
    if model_type == 'scipy':
        model = model.todense()  # FIXME: this does not work for large models. Need to calculate cosine with sparse matrix
    dim = model.shape[1]
    splits = dutils.load_kfold_splits(vocab, [dataset], kfold_size,
                                      output_logpath=None)
    for fold in splits.keys():
        results.append(evaluate(model, splits[fold]['test'], [dataset],
                                metric, distance))
    if kfold_size == 0:
        result = results[0]
        logger.info('{} = {}'.format(metric, result[0]))
    else:
        result = np.mean(results)
        logger.info('avg test {} = {}'.format(metric, result))
        logger.info('std test {} = {}'.format(metric, np.std(results)))
    logger.info('dim = {}'.format(dim))
    return result


def evaluate_distributional_space(model, vocab, dataset, metric, model_type,
                                  distance, kfold_size):
    """Evaluate a numpy model against the MEN/Simlex/Simverb datasets."""
    logger.info('Evaluating distributional space...')
    if dataset not in ['ap', 'battig', 'essli', 'men', 'simlex', 'simverb']:
        raise Exception('Unsupported dataset: {}'.format(dataset))
    if dataset in ['men', 'simlex', 'simverb']:
        return _evaluate_word_similarity(model, vocab, dataset, metric,
                                         model_type, distance, kfold_size)
    elif dataset in ['ap', 'battig', 'essli']:
        return _evaluate_concept_categorization(model, vocab, dataset)
