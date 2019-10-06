"""Dimensionality reduction through dimensionality selection."""

import os
import logging
import random
import functools
import multiprocessing
from collections import defaultdict

import numpy as np
import entropix.core.evaluator as evaluator
import entropix.utils.data as dutils
import entropix.utils.metrix as metrix

__all__ = ('Sampler')

logger = logging.getLogger(__name__)


class Sampler():

    def __init__(self, singvectors_filepath, model_type, vocab_filepath,
                 dataset, output_basepath, num_iter, shuffle, mode, rate,
                 start, end, reduce, limit, rewind, kfolding, kfold_size,
                 max_num_threads, dev_type, debug, metric, alpha, logs_dirpath,
                 distance, singvalues_filepath, sing_alpha):
        # self._model = np.load(singvectors_filepath)
        global model  # ugly hack to bypass pickling problem on forking
        global vocab
        model, vocab = dutils.load_model_and_vocab(
            singvectors_filepath, model_type, vocab_filepath,
            singvalues_filepath, sing_alpha)
        self._vocab = vocab
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
        self._distance = distance
        if logs_dirpath:
            self._logs_basepath = os.path.join(
                logs_dirpath, os.path.basename(output_basepath))
        else:
            self._logs_basepath = output_basepath
        self._logs_filepath = None
        self._splits = dutils.load_kfold_splits(
            self._vocab, self._dataset, self._kfold_size,
            self._dev_type, self._logs_basepath)

    def debug(self, keep, fold):
        train_rmse = evaluator.evaluate(
            model[:, keep], self._splits[fold]['train'],
            dataset=self._dataset, metric='rmse', distance=self._distance)
        logger.debug('train rmse = {} on fold {}'.format(train_rmse, fold))
        train_rmse_log_name = '{}.train.rmse.log'.format(self._logs_filepath)
        with open(train_rmse_log_name, 'a', encoding='utf-8') as train_rmse_log:
            print(train_rmse, file=train_rmse_log)
        train_spr = evaluator.evaluate(
            model[:, keep], self._splits[fold]['train'],
            dataset=self._dataset, metric='spr', distance=self._distance)
        logger.debug('train spr = {} on fold {}'.format(train_spr, fold))
        train_spr_log_name = '{}.train.spr.log'.format(self._logs_filepath)
        with open(train_spr_log_name, 'a', encoding='utf-8') as train_spr_log:
            print(train_spr, file=train_spr_log)
        if self._dev_type == 'regular':
            dev_rmse = evaluator.evaluate(
                model[:, keep], self._splits[fold]['dev'],
                dataset=self._dataset, metric='rmse', distance=self._distance)
            logger.debug('dev rmse = {} for fold {}'.format(dev_rmse, fold))
            dev_spr = evaluator.evaluate(
                model[:, keep], self._splits[fold]['spr'],
                dataset=self._dataset, metric='rmse', distance=self._distance)
            logger.debug('dev spr = {} for fold {}'.format(dev_spr, fold))
        if 'test' in self._splits[fold]:
            test_rmse = evaluator.evaluate(
                model[:, keep], self._splits[fold]['test'],
                dataset=self._dataset, metric='rmse', distance=self._distance)
            logger.debug('test rmse = {} for fold {}'.format(test_rmse, fold))
            test_rmse_log_name = '{}.test.rmse.log'.format(self._logs_filepath)
            with open(test_rmse_log_name, 'a', encoding='utf-8') as test_rmse_log:
                print(test_rmse, file=test_rmse_log)
            test_spr = evaluator.evaluate(
                model[:, keep], self._splits[fold]['test'],
                dataset=self._dataset, metric='spr', distance=self._distance)
            logger.debug('test spr = {} for fold {}'.format(test_spr, fold))
            test_spr_log_name = '{}.test.spr.log'.format(self._logs_filepath)
            with open(test_spr_log_name, 'a', encoding='utf-8') as test_spr_log:
                print(test_spr, file=test_spr_log)

    def display_scores(self):
        num_folds = len(self._results.keys())
        with open('{}.results'.format(self._logs_basepath), 'w', encoding='utf-8') as out_res:
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
            logger.info('-----------------------------------------')
            min_train_spr = np.min([x['train']['spr'] for x in self._results.values()])
            max_train_spr = np.max([x['train']['spr'] for x in self._results.values()])
            avg_train_spr = np.mean([x['train']['spr'] for x in self._results.values()])
            std_train_spr = np.std([x['train']['spr'] for x in self._results.values()])
            min_test_spr = np.min([x['test']['spr'] for x in self._results.values()])
            max_test_spr = np.max([x['test']['spr'] for x in self._results.values()])
            avg_test_spr = np.mean([x['test']['spr'] for x in self._results.values()])
            std_test_spr = np.std([x['test']['spr'] for x in self._results.values()])
            min_train_rmse = np.min([x['train']['rmse'] for x in self._results.values()])
            max_train_rmse = np.max([x['train']['rmse'] for x in self._results.values()])
            avg_train_rmse = np.mean([x['train']['rmse'] for x in self._results.values()])
            std_train_rmse = np.std([x['train']['rmse'] for x in self._results.values()])
            min_test_rmse = np.min([x['test']['rmse'] for x in self._results.values()])
            max_test_rmse = np.max([x['test']['rmse'] for x in self._results.values()])
            avg_test_rmse = np.mean([x['test']['rmse'] for x in self._results.values()])
            std_test_rmse = np.std([x['test']['rmse'] for x in self._results.values()])
            min_dim = np.min([x['dim'] for x in self._results.values()])
            max_dim = np.max([x['dim'] for x in self._results.values()])
            avg_dim = np.mean([x['dim'] for x in self._results.values()])
            std_dim = np.std([x['dim'] for x in self._results.values()])
            logger.info('Min train spr = {}'.format(min_train_spr))
            print('Min train spr = {}'.format(min_train_spr), file=out_res)
            logger.info('Max train spr = {}'.format(max_train_spr))
            print('Max train spr = {}'.format(max_train_spr), file=out_res)
            logger.info('Average train spr = {}'.format(avg_train_spr))
            print('Average train spr = {}'.format(avg_train_spr), file=out_res)
            logger.info('Std train spr = {}'.format(std_train_spr))
            print('Std train spr = {}'.format(std_train_spr), file=out_res)
            logger.info('Min test spr = {}'.format(min_test_spr))
            print('Min test spr = {}'.format(min_test_spr), file=out_res)
            logger.info('Max test spr = {}'.format(max_test_spr))
            print('Max test spr = {}'.format(max_test_spr), file=out_res)
            logger.info('Average test spr = {}'.format(avg_test_spr))
            print('Average test spr = {}'.format(avg_test_spr), file=out_res)
            logger.info('Std test spr = {}'.format(std_test_spr))
            print('Std test spr = {}'.format(std_test_spr), file=out_res)
            logger.info('Min train rmse = {}'.format(min_train_rmse))
            print('Min train rmse = {}'.format(min_train_rmse), file=out_res)
            logger.info('Max train rmse = {}'.format(max_train_rmse))
            print('Max train rmse = {}'.format(max_train_rmse), file=out_res)
            logger.info('Average train rmse = {}'.format(avg_train_rmse))
            print('Average train rmse = {}'.format(avg_train_rmse), file=out_res)
            logger.info('Std train rmse = {}'.format(std_train_rmse))
            print('Std train rmse = {}'.format(std_train_rmse), file=out_res)
            logger.info('Min test rmse = {}'.format(min_test_rmse))
            print('Min test rmse = {}'.format(min_test_rmse), file=out_res)
            logger.info('Max test rmse = {}'.format(max_test_rmse))
            print('Max test rmse = {}'.format(max_test_rmse), file=out_res)
            logger.info('Average test rmse = {}'.format(avg_test_rmse))
            print('Average test rmse = {}'.format(avg_test_rmse), file=out_res)
            logger.info('Std test rmse = {}'.format(std_test_rmse))
            print('Std test rmse = {}'.format(std_test_rmse), file=out_res)
            logger.info('Min dim = {}'.format(min_dim))
            print('Min dim = {}'.format(min_dim), file=out_res)
            logger.info('Max dim = {}'.format(max_dim))
            print('Max dim = {}'.format(max_dim), file=out_res)
            logger.info('Average dim = {}'.format(avg_dim))
            print('Average dim = {}'.format(avg_dim), file=out_res)
            logger.info('Std dim = {}'.format(std_dim))
            print('Std dim = {}'.format(std_dim), file=out_res)
            logger.info('One-liner: {} & {} & {} & {} & {} & {} & {} & {} & '
                        '{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} '
                        '& {}'.format(
                    min_train_spr, max_train_spr, avg_train_spr, std_train_spr,
                    min_test_spr, max_test_spr, avg_test_spr, std_test_spr,
                    min_train_rmse, max_train_rmse, avg_train_rmse, std_train_rmse,
                    min_test_rmse, max_test_rmse, avg_test_rmse, std_test_rmse,
                    min_dim, max_dim, avg_dim, std_dim))
            print('One-liner: {} & {} & {} & {} & {} & {} & {} & {} & '
                        '{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} '
                        '& {}'.format(
                    min_train_spr, max_train_spr, avg_train_spr, std_train_spr,
                    min_test_spr, max_test_spr, avg_test_spr, std_test_spr,
                    min_train_rmse, max_train_rmse, avg_train_rmse, std_train_rmse,
                    min_test_rmse, max_test_rmse, avg_test_rmse, std_test_rmse,
                    min_dim, max_dim, avg_dim, std_dim), file=out_res)

    def compute_scores(self, keep, fold):
        self._results[fold] = {
            'train': {
                'spr': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['train'],
                    self._dataset, metric='spr', distance=self._distance),
                'rmse': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['train'],
                    self._dataset, metric='rmse', distance=self._distance),
            },
            'test': {
                'spr': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['test'],
                    self._dataset, metric='spr', distance=self._distance),
                'rmse': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['test'],
                    self._dataset, metric='rmse', distance=self._distance)
            },
            'dim': len(keep)
        }
        if self._dev_type == 'regular':
            self._results[fold]['dev'] = {
                'spr': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['dev'],
                    self._dataset, metric='spr', distance=self._distance),
                'rmse': evaluator.evaluate(
                    model[:, keep], self._splits[fold]['dev'],
                    self._dataset, metric='rmse', distance=self._distance)
            }

    def reduce_dim(self, dims, best_train_eval_metric,
                   best_dev_eval_metric, iterx, step, save, fold):
        logger.info('Reducing dimensions while maintaining highest score '
                    'on eval metric {}. Step = {}. Best train eval metric = {}'
                    .format(self._metric, step, best_train_eval_metric))
        dims_set = set(dims)
        for dim_idx in dims:
            dims_set.remove(dim_idx)
            train_eval_metric = evaluator.evaluate(
                model[:, list(dims_set)], self._splits[fold]['train'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha,
                distance=self._distance)
            if evaluator.is_degrading(train_eval_metric,
                                      best_train_eval_metric,
                                      metric=self._metric):
                dims_set.add(dim_idx)
                continue
            if self._dev_type == 'regular':
                dev_eval_metric = evaluator.evaluate(
                    model[:, list(dims_set)], self._splits[fold]['dev'],
                    dataset=self._dataset, metric=self._metric,
                    alpha=self._alpha, distance=self._distance)
                if evaluator.is_degrading(dev_eval_metric,
                                          best_dev_eval_metric,
                                          metric=self._metric):
                    dims_set.add(dim_idx)
                    continue
            logger.info('Constant best train {} = {} for fold {} removing '
                        'dim_idx = {}. New ndim = {}'
                        .format(self._metric, train_eval_metric, fold,
                                dim_idx, len(dims_set)))
            best_train_eval_metric = train_eval_metric
            if self._dev_type == 'regular':
                best_dev_eval_metric = dev_eval_metric
            if self._debug:
                self.debug(list(dims_set), fold)
        reduce_filepath = '{}.keep.iter-{}.reduce.step-{}.txt'.format(
            self._output_filepath, iterx, step)
        logger.info('Finished reducing dims')
        keep = list(sorted(dims_set, key=dims.index))
        if save:
            logger.info('Saving list of reduced keep idx to {}'
                        .format(reduce_filepath))
            with open(reduce_filepath, 'w', encoding='utf-8') as reduced_stream:
                print('\n'.join([str(idx) for idx in sorted(keep)]),
                      file=reduced_stream)
        if len(keep) != len(dims):
            step += 1
            keep, best_train_eval_metric, best_dev_eval_metric =\
             self.reduce_dim(keep, best_train_eval_metric,
                             best_dev_eval_metric, iterx, step, save, fold)
        return keep, best_train_eval_metric, best_dev_eval_metric

    def increase_dim(self, keep, alldims, iterx, fold, best_train_eval_metric,
                     best_dev_eval_metric):
        logger.info('Increasing dimensions to maximize score on eval metric '
                    '{}. Iteration = {}. Best train eval metric = {}'
                    .format(self._metric, iterx, best_train_eval_metric))
        dims = [idx for idx in alldims if idx not in keep]
        added_counter = 0
        for idx, dim_idx in enumerate(dims):
            keep.append(dim_idx)
            train_eval_metric = evaluator.evaluate(
                model[:, keep], self._splits[fold]['train'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha,
                distance=self._distance)
            if evaluator.is_improving(train_eval_metric,
                                      best_train_eval_metric,
                                      metric=self._metric):
                if self._dev_type == 'regular':
                    dev_eval_metric = evaluator.evaluate(
                        model[:, keep], self._splits[fold]['dev'],
                        dataset=self._dataset, metric=self._metric,
                        alpha=self._alpha, distance=self._distance)
                    if evaluator.is_degrading(dev_eval_metric,
                                              best_dev_eval_metric,
                                              metric=self._metric):
                        keep.pop()
                        continue
                    best_dev_eval_metric = dev_eval_metric
                added_counter += 1
                best_train_eval_metric = train_eval_metric
                logger.info('New best train {} = {} on fold {} with ndim = {} '
                            'at idx = {} and dim_idx = {}'.format(
                                self._metric, best_train_eval_metric, fold,
                                len(keep), idx, dim_idx))
                if self._debug:
                    self.debug(keep, fold)
                if self._mode == 'mix' and added_counter % self._rate == 0:
                    keep, best_train_eval_metric, best_dev_eval_metric =\
                     self.reduce_dim(keep, best_train_eval_metric,
                                     best_dev_eval_metric, iterx, step=1,
                                     save=False, fold=fold)
            else:
                keep.pop()
        return keep, best_train_eval_metric, best_dev_eval_metric

    def sample_seq_mix(self, keep, alldims, fold):
        logger.info('Shuffling mode {}'
                    .format('ON' if self._shuffle else 'OFF'))
        logger.info('Iterating over {} dims starting at {} and ending at {}'
                    .format(model.shape[1], self._start, self._end))
        if self._shuffle:  # first iteration
            keep = np.random.choice(
                list(range(model.shape[1]))[self._start:self._end],
                size=2, replace=False).tolist()
        else:
            keep = [self._start, self._start+1]  # start at 2-dims
        best_train_eval_metric = evaluator.evaluate(
            model[:, keep], self._splits[fold]['train'], dataset=self._dataset,
            metric=self._metric, alpha=self._alpha, distance=self._distance)
        logger.debug('Initial train eval metric = {}'
                     .format(best_train_eval_metric))
        if self._dev_type == 'nodev':
            best_dev_eval_metric = 0
        elif self._dev_type == 'regular':
            best_dev_eval_metric = evaluator.evaluate(
                model[:, keep], self._splits[fold]['dev'],
                dataset=self._dataset, metric=self._metric, alpha=self._alpha,
                distance=self._distance)
        for iterx in range(1, self._num_iter+1):
            keep_filepath = '{}.keep.iter-{}.txt'.format(
                self._output_filepath, iterx)
            keep, best_train_eval_metric, best_dev_eval_metric = self.increase_dim(
                keep, alldims, iterx, fold, best_train_eval_metric,
                best_dev_eval_metric)
            logger.info('Finished dim increase. Saving list of keep idx to {}'
                        .format(keep_filepath))
            with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
                print('\n'.join([str(idx) for idx in sorted(keep)]),
                      file=keep_stream)
            if self._reduce:
                keep, best_train_eval_metric, best_dev_eval_metric =\
                 self.reduce_dim(keep, best_train_eval_metric,
                                 best_dev_eval_metric, iterx, step=1,
                                 save=True, fold=fold)
        return fold, keep

    def rewind(self, dims, alldims, fold, best_metric, least_worst_metric,
               max_num_dim_best):
        for i in dims:
            best_dim_idx = -1
            least_worst_dim = -1
            tmp_dims = [dim for dim in dims if dim != i]
            for idx in alldims:
                tmp_dims.append(idx)
                eval_metric = evaluator.evaluate(
                    model[:, list(tmp_dims)],
                    self._splits[fold]['train'], dataset=self._dataset,
                    metric=self._metric,
                    alpha=self._alpha, distance=self._distance)
                if evaluator.is_improving(eval_metric,
                                          best_metric,
                                          metric=self._metric):
                    best_metric = eval_metric
                    best_dim_idx = idx
                elif evaluator.is_improving(eval_metric,
                                            least_worst_metric,
                                            metric=self._metric):
                    least_worst_metric = eval_metric
                    least_worst_dim = idx
                tmp_dims.pop()
            if best_dim_idx == -1:
                if least_worst_dim == -1:
                    least_worst_dim = i
                logger.info('Could not find a better metric when rewinding '
                            'dim={}. Added least worst dim={} to continue'
                            .format(i, least_worst_dim))
                tmp_dims.append(least_worst_dim)
            else:
                tmp_dims.append(best_dim_idx)
                max_num_dim_best = len(tmp_dims)
                logger.info(
                    'Rewind best {} = {} on fold {} with dims = '
                    '{}'.format(self._metric, best_metric,
                                fold, tmp_dims))
        return tmp_dims, best_metric, least_worst_metric, max_num_dim_best

    def sample_limit(self, alldims, fold):
        """Increase dims up to dlim taking the best dim each time.

        With rewind, go back each time new dim is added to check if can add a
        better one.
        """
        best_metric = metrix.init_eval_metrix(self._metric, self._alpha)
        dims = []  # dims shouldn't exceed 100 so this should be ok. Limit mode is intended for compact models primarly
        max_num_dim_best = 0
        for k in range(self._limit):
            best_dim_idx = -1
            least_worst_dim = -1
            least_worst_metric = metrix.init_eval_metrix(self._metric,
                                                         self._alpha)
            for dim_idx in alldims:
                if dim_idx in dims:
                    continue
                dims.append(dim_idx)
                eval_metric = evaluator.evaluate(
                    model[:, list(dims)], self._splits[fold]['train'],
                    dataset=self._dataset, metric=self._metric,
                    alpha=self._alpha, distance=self._distance)
                if evaluator.is_improving(eval_metric,
                                          best_metric,
                                          metric=self._metric):
                    best_metric = eval_metric
                    best_dim_idx = dim_idx
                elif evaluator.is_improving(eval_metric,
                                            least_worst_metric,
                                            metric=self._metric):
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
                logger.info('Current best {} = {} on fold {} with dims = '
                            '{}'.format(self._metric, best_metric,
                                        fold, dims))
            if self._rewind and len(dims) > 1:
                dims, best_metric, least_worst_metric, max_num_dim_best =\
                 self.rewind(dims, alldims, fold, best_metric,
                             least_worst_metric, max_num_dim_best)
        logger.info('Best eval metric = {} found using {} dims = {}'
                    .format(best_metric, max_num_dim_best, sorted(dims)))
        if self._rewind:
            final_filepath = '{}.final.rewind.txt'.format(
                self._output_filepath)
        else:
            final_filepath = '{}.final.txt'.format(self._output_filepath)
        logger.info('Saving output to file {}'.format(final_filepath))
        with open(final_filepath, 'w', encoding='utf-8') as final_stream:
            print('\n'.join([str(idx) for idx in sorted(dims)]),
                  file=final_stream)
        return fold, dims

    def sample_with_kfold(self, keep, dims, fold):
        self._output_filepath = '{}.kfold{}-{}'.format(
            self._output_basepath, fold, len(self._splits.keys()))
        if self._debug:
            self._logs_filepath = '{}.kfold{}-{}'.format(
                self._logs_basepath, fold, len(self._splits.keys()))
        if self._mode == 'limit':
            return self.sample_limit(dims, fold)
        return self.sample_seq_mix(keep, dims, fold)


    def sample_dimensions(self):
        logger.info('Sampling dimensions over a total of {} dims, optimizing '
                    'on {} using {} mode...'
                    .format(model.shape[1], self._dataset, self._mode))
        if self._mode not in ['seq', 'mix', 'limit']:
            raise Exception('Unsupported mode {}'.format(self._mode))
        if not self._kfolding:
            raise Exception('Non-kfold mode needs reimplementation')
        if self._end > model.shape[1]:
            raise Exception('End parameter is > model.shape[1]: {} > {}'
                            .format(self._end, model.shape[1]))
        if self._end == 0:
            self._end = model.shape[1]
        if self._dataset not in ['men', 'simlex', 'simverb', 'sts2012']:
            raise Exception('Unsupported eval dataset: {}'
                            .format(self._dataset))
        alldims = list(range(model.shape[1]))[self._start:self._end]
        if self._shuffle:
            random.shuffle(alldims)
        if self._kfolding:
            # sample dimensons multi-threaded on all kfolds
            num_folds = len(self._splits.keys())
            logger.info('Applying kfolding on k={} folds where each test '
                        'fold is of size {} and accounts for {}% of '
                        'the data'.format(
                            num_folds,
                            len(self._splits[1]['test']['sim']),
                            self._kfold_size*100))
            num_threads = num_folds if num_folds <= self._max_num_threads \
                else self._max_num_threads
            with multiprocessing.Pool(num_threads) as pool:
                _sample_with_kfold = functools.partial(
                    self.sample_with_kfold, [], alldims)
                for fold, keep in pool.imap_unordered(_sample_with_kfold,
                                                      range(1,
                                                            num_folds+1)):
                    self.compute_scores(keep, fold)
                self.display_scores()
