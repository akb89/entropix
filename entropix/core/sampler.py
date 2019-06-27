"""Dimensionality reduction through dimensionality selection."""

import os
import logging
import random
import functools
import multiprocessing
from collections import defaultdict

import numpy as np
import entropix.core.evaluator as evaluator

__all__ = ('Sampler')

logger = logging.getLogger(__name__)


# TODO: refactor this
def sample_limit(model, dataset, left_idx, right_idx, sim, output_filepath,
                 limit, start, end, rewind):
    """Increase dims up to dlim taking the best dim each time.

    With rewind, go back each time new dim is added to check if can add a
    better one.
    """
    max_spr = 0.
    dims = set()
    max_num_dim_best = 0
    for k in range(limit):
        best_dim_idx = -1
        least_worst_dim = -1
        least_worst_spr = 0.
        for dim_idx in range(start, end):
            if dim_idx in dims:
                continue
            dims.add(dim_idx)
            spr = evaluator.evaluate(model[:, list(dims)], left_idx, right_idx,
                                     sim, dataset)
            if spr > max_spr:
                max_spr = spr
                best_dim_idx = dim_idx
                logger.info('Current best = {} with dims = {}'
                            .format(max_spr, sorted(dims)))
            elif spr > least_worst_spr:
                least_worst_spr = spr
                least_worst_dim = dim_idx
            dims.remove(dim_idx)
        if best_dim_idx == -1:
            logger.info('Could not find a better SPR correlation with {} '
                        'dims. Added least worst dim to continue'.format(k+1))
            dims.add(least_worst_dim)
        else:
            dims.add(best_dim_idx)
            max_num_dim_best += 1
        if rewind and len(dims) > 1:
            for i in sorted(dims):
                best_dim = -1
                if i == least_worst_dim or i == best_dim_idx:
                    continue
                dims.remove(i)
                for idx in range(start, end):
                    if idx == i or idx in dims:
                        continue
                    dims.add(idx)
                    spr = evaluator.evaluate(model[:, list(dims)], left_idx,
                                             right_idx, sim, dataset)
                    if spr > max_spr:
                        max_spr = spr
                        best_dim = idx
                        logger.info('Rewinded best = {} with dims = {}'
                                    .format(max_spr, sorted(dims)))
                    dims.remove(idx)
                if best_dim == -1:
                    dims.add(i)
                else:
                    dims.add(best_dim)
    logger.info('Best SPR = {} found using {} dims = {}'
                .format(max_spr, max_num_dim_best, sorted(dims)))
    if rewind:
        final_filepath = '{}.final.rewind.txt'.format(output_basename)
    else:
        final_filepath = '{}.final.txt'.format(output_basename)
    logger.info('Saving output to file {}'.format(final_filepath))
    with open(final_filepath, 'w', encoding='utf-8') as final_stream:
        print('\n'.join([str(idx) for idx in sorted(dims)]), file=final_stream)


class Sampler():

    def __init__(self, singvectors_filepath, vocab_filepath, dataset,
                 output_basepath, num_iter, shuffle, mode, rate, start, end,
                 reduce, limit, rewind, kfolding, kfold_size, max_num_threads,
                 dev_type, debug, metric, alpha, logs_dirpath):
        #self._model = np.load(singvectors_filepath)
        global model
        logger.info('Loading numpy model...')
        model = np.load(singvectors_filepath)
        self._vocab_filepath = vocab_filepath
        self._dataset = dataset
        self._output_basepath = output_basepath
        self._output_filepath = None
        self._num_iter = num_iter
        self._shuffle = shuffle
        self._mode = mode
        self._rate = rate
        self._start = start
        self._end = end
        self._reduce = reduce
        self._limit = limit
        self._rewind = rewind
        self._kfolding = kfolding
        self._kfold_size = kfold_size
        self._max_num_threads = max_num_threads
        self._dev_type = dev_type
        self._debug = debug
        self._results = defaultdict(defaultdict)
        self._metric = metric
        self._alpha = alpha
        if logs_dirpath:
            self._logs_basepath = os.path.join(
                logs_dirpath, os.path.basename(output_basepath))
        else:
            self._logs_basepath = output_basepath
        self._logs_filepath = None

    def debug(self, keep, splits, fold):
        train_rmse = evaluator.get_eval_metric(
            model[:, list(keep)], splits[fold]['train'],
            dataset=self._dataset, metric='rmse')
        logger.debug('train rmse = {} on fold {}'.format(train_rmse, fold))
        train_rmse_log_name = '{}.train.rmse.log'.format(self._logs_filepath)
        with open(train_rmse_log_name, 'a') as train_rmse_log:
            print(train_rmse, file=train_rmse_log)
        train_spr = evaluator.get_eval_metric(
            model[:, list(keep)], splits[fold]['train'],
            dataset=self._dataset, metric='spr')
        logger.debug('train spr = {} on fold {}'.format(train_spr, fold))
        train_spr_log_name = '{}.train.spr.log'.format(self._logs_filepath)
        with open(train_spr_log_name, 'a') as train_spr_log:
            print(train_spr, file=train_spr_log)
        if self._dev_type == 'regular':
            dev_rmse = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['dev'],
                dataset=self._dataset, metric='rmse')
            logger.debug('dev rmse = {} for fold {}'.format(dev_rmse, fold))
            dev_spr = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['spr'],
                dataset=self._dataset, metric='rmse')
            logger.debug('dev spr = {} for fold {}'.format(dev_spr, fold))
        if 'test' in splits[fold]:
            test_rmse = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['test'],
                dataset=self._dataset, metric='rmse')
            logger.debug('test rmse = {} for fold {}'.format(test_rmse, fold))
            test_rmse_log_name = '{}.test.rmse.log'.format(self._logs_filepath)
            with open(test_rmse_log_name, 'a') as test_rmse_log:
                print(test_rmse, file=test_rmse_log)
            test_spr = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['test'],
                dataset=self._dataset, metric='spr')
            logger.debug('test spr = {} for fold {}'.format(test_spr, fold))
            test_spr_log_name = '{}.test.spr.log'.format(self._logs_filepath)
            with open(test_spr_log_name, 'a') as test_spr_log:
                print(test_spr, file=test_spr_log)


    def display_scores(self):
        num_folds = len(self._results.keys())
        with open('{}.results'.format(self._logs_basepath), 'w') as out_res:
            for fold, values, in sorted(self._results.items()):
                logger.info('Fold {}/{} dim = {}'.format(
                    fold, num_folds, values['dim']))
                print('Fold {}/{} dim = {}'.format(
                    fold, num_folds, values['dim']), file=out_res)
                logger.info('Fold {}/{} train spr = {}'.format(
                    fold, num_folds, values['train']['spr']))
                print('Fold {}/{} train spr = {}'.format(
                    fold, num_folds, values['train']['spr']), file=out_res)
                logger.info('Fold {}/{} train rmse = {}'.format(
                    fold, num_folds, values['train']['rmse']))
                print('Fold {}/{} train rmse = {}'.format(
                    fold, num_folds, values['train']['rmse']), file=out_res)
                logger.info('Fold {}/{} test spr = {}'.format(
                    fold, num_folds, values['test']['spr']))
                print('Fold {}/{} test spr = {}'.format(
                    fold, num_folds, values['test']['spr']), file=out_res)
                logger.info('Fold {}/{} test rmse = {}'.format(
                    fold, num_folds, values['test']['rmse']))
                print('Fold {}/{} test rmse = {}'.format(
                    fold, num_folds, values['test']['rmse']), file=out_res)
            logger.info('--------------------')
            logger.info('Min dim = {}'.format(
                np.min([x['dim'] for x in self._results.values()])))
            print('Min dim = {}'.format(
                np.min([x['dim'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Max dim = {}'.format(
                np.max([x['dim'] for x in self._results.values()])))
            print('Max dim = {}'.format(
                np.max([x['dim'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Average dim = {}'.format(
                np.mean([x['dim'] for x in self._results.values()])))
            print('Average dim = {}'.format(
                np.mean([x['dim'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Std dim = {}'.format(
                np.std([x['dim'] for x in self._results.values()])))
            print('Std dim = {}'.format(
                np.std([x['dim'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Min train spr = {}'.format(
                np.min([x['train']['spr'] for x in self._results.values()])))
            print('Min train spr = {}'.format(
                np.min([x['train']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Max train spr = {}'.format(
                np.max([x['train']['spr'] for x in self._results.values()])))
            print('Max train spr = {}'.format(
                np.max([x['train']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Average train spr = {}'.format(
                np.mean([x['train']['spr'] for x in self._results.values()])))
            print('Average train spr = {}'.format(
                np.mean([x['train']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Std train spr = {}'.format(
                np.std([x['train']['spr'] for x in self._results.values()])))
            print('Std train spr = {}'.format(
                np.std([x['train']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Min train rmse = {}'.format(
                np.min([x['train']['rmse'] for x in self._results.values()])))
            print('Min train rmse = {}'.format(
                np.min([x['train']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Max train rmse = {}'.format(
                np.max([x['train']['rmse'] for x in self._results.values()])))
            print('Max train rmse = {}'.format(
                np.max([x['train']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Average train rmse = {}'.format(
                np.mean([x['train']['rmse'] for x in self._results.values()])))
            print('Average train rmse = {}'.format(
                np.mean([x['train']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Std train rmse = {}'.format(
                np.std([x['train']['rmse'] for x in self._results.values()])))
            print('Std train rmse = {}'.format(
                np.std([x['train']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Min test spr = {}'.format(
                np.min([x['test']['spr'] for x in self._results.values()])))
            print('Min test spr = {}'.format(
                np.min([x['test']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Max test spr = {}'.format(
                np.max([x['test']['spr'] for x in self._results.values()])))
            print('Max test spr = {}'.format(
                np.max([x['test']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Average test spr = {}'.format(
                np.mean([x['test']['spr'] for x in self._results.values()])))
            print('Average test spr = {}'.format(
                np.mean([x['test']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Std test spr = {}'.format(
                np.std([x['test']['spr'] for x in self._results.values()])))
            print('Std test spr = {}'.format(
                np.std([x['test']['spr'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Min test rmse = {}'.format(
                np.min([x['test']['rmse'] for x in self._results.values()])))
            print('Min test rmse = {}'.format(
                np.min([x['test']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Max test rmse = {}'.format(
                np.max([x['test']['rmse'] for x in self._results.values()])))
            print('Max test rmse = {}'.format(
                np.max([x['test']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Average test rmse = {}'.format(
                np.mean([x['test']['rmse'] for x in self._results.values()])))
            print('Average test rmse = {}'.format(
                np.mean([x['test']['rmse'] for x in self._results.values()])),
                  file=out_res)
            logger.info('Std test rmse = {}'.format(
                np.std([x['test']['rmse'] for x in self._results.values()])))
            print('Std test rmse = {}'.format(
                np.std([x['test']['rmse'] for x in self._results.values()])),
                  file=out_res)

    def compute_scores(self, splits, keep, fold):
        self._results[fold] = {
            'train': {
                'spr': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['train'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['train'],
                    self._dataset, metric='rmse'),
            },
            'test': {
                'spr': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['test'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['test'],
                    self._dataset, metric='rmse')
            },
            'dim': len(keep)
        }
        if self._dev_type == 'regular':
            self._results[fold]['dev'] = {
                'spr': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['dev'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.get_eval_metric(
                    model[:, list(keep)], splits[fold]['dev'],
                    self._dataset, metric='rmse')
            }


    def reduce_dim(self, keep, splits, best_train_eval_metric,
                   best_dev_eval_metric, iterx, step, save, fold):
        logger.info('Reducing dimensions while maintaining highest score '
                    'on eval metric {}. Step = {}'.format(self._metric, step))
        remove = set()
        dim_indexes = sorted(keep)  # make sure to always iterate over dims
                                    # in a similar way
        for dim_idx in dim_indexes:
            dims = keep.difference(remove)
            dims.remove(dim_idx)
            train_eval_metric = evaluator.get_eval_metric(
                model[:, list(dims)], splits[fold]['train'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha)
            if evaluator.is_degrading(train_eval_metric,
                                      best_train_eval_metric,
                                      metric=self._metric):
                continue
            if self._dev_type == 'regular':
                dev_eval_metric = evaluator.get_eval_metric(
                    model[:, list(dims)], splits[fold]['dev'],
                    dataset=self._dataset, metric=self._metric,
                    alpha=self._alpha)
                if evaluator.is_degrading(dev_eval_metric,
                                          best_dev_eval_metric,
                                          metric=self._metric):
                    continue
            remove.add(dim_idx)
            logger.info('Constant best train {} = {} for fold {} removing '
                        'dim_idx = {}. New ndim = {}'
                        .format(self._metric, train_eval_metric, fold,
                                dim_idx, len(dims)))
            best_train_eval_metric = train_eval_metric
            if self._dev_type == 'regular':
                best_dev_eval_metric = dev_eval_metric
            if self._debug:
                self.debug(keep, splits, fold)
        reduce_filepath = '{}.keep.iter-{}.reduce.step-{}.txt'.format(
            self._output_filepath, iterx, step)
        logger.info('Finished reducing dims')
        keep = keep.difference(remove)
        if save:
            logger.info('Saving list of reduced keep idx to {}'
                        .format(reduce_filepath))
            with open(reduce_filepath, 'w', encoding='utf-8') as reduced_stream:
                print('\n'.join([str(idx) for idx in sorted(keep)]),
                      file=reduced_stream)
        if remove:
            step += 1
            self.reduce_dim(keep, splits, best_train_eval_metric,
                            best_dev_eval_metric, iterx, step, save, fold)
        return keep

    def increase_dim(self, keep, dims, splits, iterx, fold):
        logger.info('Increasing dimensions to maximize score on eval metric '
                    '{}. Iteration = {}'.format(self._metric, iterx))
        best_train_eval_metric = evaluator.get_eval_metric(
            model[:, list(keep)], splits[fold]['train'], dataset=self._dataset,
            metric=self._metric, alpha=self._alpha)
        added_counter = 0
        if self._dev_type == 'nodev':
            best_dev_eval_metric = 0
        elif self._dev_type == 'regular':
            best_dev_eval_metric = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['dev'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha)
        for idx, dim_idx in enumerate(dims):
            keep.add(dim_idx)
            train_eval_metric = evaluator.get_eval_metric(
                model[:, list(keep)], splits[fold]['train'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha)
            if evaluator.is_improving(train_eval_metric,
                                      best_train_eval_metric,
                                      metric=self._metric):
                if self._dev_type == 'regular':
                    dev_eval_metric = evaluator.get_eval_metric(
                        model[:, list(keep)], splits[fold]['dev'],
                        dataset=self._dataset, metric=self._metric,
                        alpha=self._alpha)
                    if evaluator.is_degrading(dev_eval_metric,
                                              best_dev_eval_metric,
                                              metric=self._metric):
                        keep.remove(dim_idx)
                        continue
                    best_dev_eval_metric = dev_eval_metric
                added_counter += 1
                best_train_eval_metric = train_eval_metric
                logger.info('New best train {} = {} on fold {} with ndim = {} '
                            'at idx = {} and dim_idx = {}'.format(
                                self._metric, best_train_eval_metric, fold,
                                len(keep), idx, dim_idx))
                if self._debug:
                    self.debug(keep, splits, fold)
                if self._mode == 'mix' and added_counter % self._rate == 0:
                    keep = self.reduce_dim(keep, splits,
                                           best_train_eval_metric,
                                           best_dev_eval_metric, iterx, step=1,
                                           save=False, fold=fold)
            else:
                keep.remove(dim_idx)
        return keep, best_train_eval_metric, best_dev_eval_metric

    def sample_seq_mix(self, splits, keep, dims, fold):
        logger.info('Shuffling mode {}'
                    .format('ON' if self._shuffle else 'OFF'))
        logger.info('Iterating over {} dims starting at {} and ending at {}'
                    .format(model.shape[1], self._start, self._end))
        for iterx in range(1, self._num_iter+1):
            keep_filepath = '{}.keep.iter-{}.txt'.format(
                self._output_filepath, iterx)
            keep, best_train_eval_metric, best_dev_eval_metric = self.increase_dim(
                keep, dims, splits, iterx, fold)
            logger.info('Finished dim increase. Saving list of keep idx to {}'
                        .format(keep_filepath))
            with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
                print('\n'.join([str(idx) for idx in sorted(keep)]),
                      file=keep_stream)
            if self._reduce:
                keep = self.reduce_dim(keep, splits, best_train_eval_metric,
                                       best_dev_eval_metric, iterx, step=1,
                                       save=True, fold=fold)
            return fold, keep

    def sample_seq_mix_with_kfold(self, splits, keep, dims, fold):
        self._output_filepath = '{}.kfold{}-{}'.format(
            self._output_basepath, fold, len(splits.keys()))
        if self._debug:
            self._logs_filepath = '{}.kfold{}-{}'.format(
                self._logs_basepath, fold, len(splits.keys()))
        return self.sample_seq_mix(splits, keep, dims, fold)

    def sample_dimensions(self):
        logger.info('Sampling dimensions over a total of {} dims, optimizing '
                    'on {} using {} mode...'
                    .format(model.shape[1], self._dataset, self._mode))
        if self._mode not in ['seq', 'mix', 'limit']:
            raise Exception('Unsupported mode {}'.format(self._mode))
        if self._end > model.shape[1]:
            raise Exception('End parameter is > model.shape[1]: {} > {}'
                            .format(self._end, model.shape[1]))
        if self._end == 0:
            self._end = model.shape[1]
        if self._dataset not in ['men', 'simlex', 'simverb', 'sts2012']:
            raise Exception('Unsupported eval dataset: {}'
                            .format(self._dataset))
        if self._kfolding:
            splits = evaluator.load_kfold_splits_dict(
                self._vocab_filepath, self._dataset, self._kfold_size,
                self._dev_type)
        else:
            vocab = evaluator.load_vocab(self._vocab_filepath)
            left_idx, right_idx, sim = evaluator.load_words_and_sim(
                vocab, self._dataset, shuffle=False)
        if self._shuffle:
            keep = set(np.random.choice(
                list(range(model.shape[1]))[self._start:self._end],
                size=2, replace=False))
        else:
            keep = set([self._start, self._start+1])  # start at 2-dims
        dims = [idx for idx in list(range(model.shape[1]))[self._start:self._end]
                if idx not in keep]
        if self._shuffle:
            random.shuffle(dims)
        if self._mode in ['seq', 'mix']:
            if self._kfolding:
                # sample dimensons multi-threaded on all kfolds
                num_folds = len(splits.keys())
                logger.info('Applying kfolding on k={} folds where each test '
                            'fold is of size {} and accounts for {}% of '
                            'the data'.format(
                                num_folds,
                                len(splits[1]['test']['sim']),
                                self._kfold_size*100))
                num_threads = num_folds if num_folds <= self._max_num_threads \
                    else self._max_num_threads
                with multiprocessing.Pool(num_threads) as pool:
                    _sample_seq_mix = functools.partial(
                        self.sample_seq_mix_with_kfold, splits, keep, dims)
                    for fold, keep in pool.imap_unordered(_sample_seq_mix,
                                                          range(1,
                                                                num_folds+1)):
                        self.compute_scores(splits, keep, fold)
                    self.display_scores()
            else:
                self._output_filepath = '{}.kfold1-1'.format(
                    self._output_basepath)
                # train with a single fold and not dev/test -> use all data
                # for dimensionality selection
                splits = {
                    1: {
                        'train': {
                            'left_idx': left_idx,
                            'right_idx': right_idx,
                            'sim': sim
                        }
                    }
                }
                self.sample_seq_mix(splits, keep, dims, fold=1)
        if self._mode == 'limit':
            sample_limit(model, dataset, left_idx, right_idx, sim,
                         output_basename, limit, start, end, rewind)
