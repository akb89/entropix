"""Dimensionality reduction through dimensionality selection."""

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
def sample_limit(model, dataset, left_idx, right_idx, sim, output_basename,
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
                 output_basename, num_iter, shuffle, mode, rate, start, end,
                 reduce, limit, rewind, kfolding, kfold_size, max_num_threads,
                 dev_type, debug):
        #self._model = np.load(singvectors_filepath)
        global model
        model = np.load(singvectors_filepath)
        self._vocab_filepath = vocab_filepath
        self._dataset = dataset
        self._output_basename = output_basename
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

    def display_scores(self):
        num_folds = len(self._results.keys())
        avg_train_spr = avg_train_rmse = avg_test_spr = avg_test_rmse = 0
        for fold, values, in sorted(self._results.items()):
            avg_train_spr += values['train']['spr']
            avg_train_rmse += values['train']['rmse']
            avg_test_spr += values['test']['spr']
            avg_test_rmse += values['test']['rmse']
            logger.info('Fold {}#{} train spr = {}'.format(
                fold, num_folds, values['train']['spr']))
            logger.info('Fold {}#{} train rmse = {}'.format(
                fold, num_folds, values['train']['rmse']))
            logger.info('Fold {}#{} test spr = {}'.format(
                fold, num_folds, values['test']['spr']))
            logger.info('Fold {}#{} test rmse = {}'.format(
                fold, num_folds, values['test']['rmse']))
        logger.info('Average train spr = {}'.format(avg_train_spr/num_folds))
        logger.info('Average train rmse = {}'.format(avg_train_rmse/num_folds))
        logger.info('Average test spr = {}'.format(avg_test_spr/num_folds))
        logger.info('Average test rmse = {}'.format(avg_test_rmse/num_folds))

    def compute_scores(self, splits, keep, fold):
        self._results[fold] = {
            'train': {
                'spr': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['train']['left_idx'],
                    splits[fold]['train']['right_idx'],
                    splits[fold]['train']['sim'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['train']['left_idx'],
                    splits[fold]['train']['right_idx'],
                    splits[fold]['train']['sim'],
                    self._dataset, metric='rmse'),
            },
            'test': {
                'spr': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['test']['left_idx'],
                    splits[fold]['test']['right_idx'],
                    splits[fold]['test']['sim'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['test']['left_idx'],
                    splits[fold]['test']['right_idx'],
                    splits[fold]['test']['sim'],
                    self._dataset, metric='rmse')
            }
        }
        if self._dev_type == 'regular':
            self._results[fold]['dev'] = {
                'spr': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['dev']['left_idx'],
                    splits[fold]['dev']['right_idx'],
                    splits[fold]['dev']['sim'],
                    self._dataset, metric='spr'),
                'rmse': evaluator.evaluate(
                    model[:, list(keep)], splits[fold]['dev']['left_idx'],
                    splits[fold]['dev']['right_idx'],
                    splits[fold]['dev']['sim'],
                    self._dataset, metric='rmse')
            }


    def reduce_dim(self, keep, splits, max_train_spr, max_dev_spr, iterx, step,
                   save, fold):
        logger.info('Reducing dimensions while maintaining highest score. '
                    'Step = {}'.format(step))
        remove = set()
        if self._shuffle:
            dim_indexes = list(keep)
            random.shuffle(dim_indexes)
        else:
            dim_indexes = sorted(keep)
        for dim_idx in dim_indexes:
            dims = keep.difference(remove)
            dims.remove(dim_idx)
            train_spr = evaluator.evaluate(
                model[:, list(dims)], splits[fold]['train']['left_idx'],
                splits[fold]['train']['right_idx'],
                splits[fold]['train']['sim'], self._dataset)
            if train_spr >= max_train_spr:
                if self._dev_type == 'regular':
                    dev_spr = evaluator.evaluate(
                        model[:, list(dims)], splits[fold]['dev']['left_idx'],
                        splits[fold]['dev']['right_idx'],
                        splits[fold]['dev']['sim'], self._dataset)
                    if dev_spr < max_dev_spr:
                        continue
                remove.add(dim_idx)
                logger.info('Constant max train spr = {} for fold {} removing '
                            'dim_idx = {}. New ndim = {}'
                            .format(max_train_spr, fold, dim_idx, len(dims)))
                if self._debug:
                    if self._dev_type == 'regular':
                        logger.debug('dev spr = {} for fold {}'.format(
                            max_dev_spr, fold))
                    if splits[fold]['test']:
                        test_spr = evaluator.evaluate(
                            model[:, list(keep)],
                            splits[fold]['test']['left_idx'],
                            splits[fold]['test']['right_idx'],
                            splits[fold]['test']['sim'], self._dataset)
                        logger.debug('test spr = {} for fold {}'.format(
                            test_spr, fold))
        if self._shuffle:
            reduce_filepath = '{}.keep.shuffled.iter-{}.reduce.step-{}.txt'.format(
                self._output_basename, iterx, step)
        else:
            reduce_filepath = '{}.keep.iter-{}.reduce.step-{}.txt'.format(
                self._output_basename, iterx, step)
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
            self.reduce_dim(keep, splits, max_train_spr, max_dev_spr, iterx,
                            step, save, fold)
        return keep

    def increase_dim(self, keep, dims, splits, iterx, fold):
        logger.info('Increasing dimensions to maximize score. Iteration = {}'
                    .format(iterx))
        max_train_spr = evaluator.evaluate(
            model[:, list(keep)], splits[fold]['train']['left_idx'],
            splits[fold]['train']['right_idx'], splits[fold]['train']['sim'],
            self._dataset)
        added_counter = 0
        if self._dev_type == 'nodev':
            max_dev_spr = 0
        elif self._dev_type == 'regular':
            max_dev_spr = evaluator.evaluate(
                model[:, list(keep)], splits[fold]['dev']['left_idx'],
                splits[fold]['dev']['right_idx'], splits[fold]['dev']['sim'],
                self._dataset)
        for idx, dim_idx in enumerate(dims):
            keep.add(dim_idx)
            train_spr = evaluator.evaluate(
                model[:, list(keep)], splits[fold]['train']['left_idx'],
                splits[fold]['train']['right_idx'],
                splits[fold]['train']['sim'], self._dataset)
            if train_spr > max_train_spr:
                if self._dev_type == 'regular':
                    dev_spr = evaluator.evaluate(
                        model[:, list(keep)], splits[fold]['dev']['left_idx'],
                        splits[fold]['dev']['right_idx'],
                        splits[fold]['dev']['sim'], self._dataset)
                    if dev_spr <= max_dev_spr:
                        keep.remove(dim_idx)
                        continue
                    max_dev_spr = dev_spr
                added_counter += 1
                max_train_spr = train_spr
                logger.info('New max train spr = {} on fold {} with ndim = {} '
                            'at idx = {} and dim_idx = {}'.format(
                                max_train_spr, fold, len(keep), idx, dim_idx))
                if self._debug:
                    if self._dev_type == 'regular':
                        logger.debug('dev spr = {} for fold {}'.format(
                            max_dev_spr, fold))
                    if splits[fold]['test']:
                        test_spr = evaluator.evaluate(
                            model[:, list(keep)],
                            splits[fold]['test']['left_idx'],
                            splits[fold]['test']['right_idx'],
                            splits[fold]['test']['sim'], self._dataset)
                        logger.debug('test spr = {} for fold {}'.format(
                            test_spr, fold))
                if self._mode == 'mix' and added_counter % self._rate == 0:
                    keep = self.reduce_dim(keep, splits, max_train_spr,
                                           max_dev_spr, iterx, step=1,
                                           save=False, fold=fold)
            else:
                keep.remove(dim_idx)
        return keep, max_train_spr, max_dev_spr

    def sample_seq_mix(self, splits, fold):
        logger.info('Shuffling mode {}'
                    .format('ON' if self._shuffle else 'OFF'))
        logger.info('Iterating over {} dims starting at {} and ending at {}'
                    .format(model.shape[1], self._start, self._end))
        if self._shuffle:
            keep = set(np.random.choice(
                list(range(model.shape[1]))[self._start:self._end],
                size=2, replace=False))
        else:
            keep = set([self._start, self._start+1])  # start at 2-dims
        for iterx in range(1, self._num_iter+1):
            dims = [idx for idx in
                    list(range(model.shape[1]))[self._start:self._end]
                    if idx not in keep]
            if self._shuffle:
                random.shuffle(dims)
                keep_filepath = '{}.keep.shuffled.iter-{}.txt'.format(
                    self._output_basename, iterx)
            else:
                keep_filepath = '{}.keep.iter-{}.txt'.format(
                    self._output_basename, iterx)
            keep, max_train_spr, max_dev_spr = self.increase_dim(
                keep, dims, splits, iterx, fold)
            logger.info('Finished dim increase. Saving list of keep idx to {}'
                        .format(keep_filepath))
            with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
                print('\n'.join([str(idx) for idx in sorted(keep)]),
                      file=keep_stream)
            if self._reduce:
                keep = self.reduce_dim(keep, splits, max_train_spr,
                                       max_dev_spr, iterx, step=1, save=True,
                                       fold=fold)
            return fold, keep, max_train_spr, max_dev_spr

    def sample_seq_mix_with_kfold(self, splits, fold):
        self._output_basename = '{}.kfold-{}#{}'.format(
            self._output_basename, fold, len(splits.keys()))
        return self.sample_seq_mix(splits, fold)

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
            left_idx, right_idx, sim = evaluator.load_words_and_sim(
                self._vocab_filepath, self._dataset, shuffle=False)
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
                        self.sample_seq_mix_with_kfold, splits)
                    for fold, keep, max_train_spr, max_dev_spr in pool.imap_unordered(
                     _sample_seq_mix, range(1, num_folds+1)):
                        # get scores on each kfold test set
                        self.compute_scores(splits, keep, fold)
                    self.display_scores()
            else:
                self._output_basename = '{}.kfold-1#1'.format(
                    self._output_basename)
                self.sample_seq_mix(splits, fold=1)
        if self._mode == 'limit':
            sample_limit(model, dataset, left_idx, right_idx, sim,
                         output_basename, limit, start, end, rewind)
