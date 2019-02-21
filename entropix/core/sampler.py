"""Dimensionality reduction through dimensionality selection."""

import logging

import numpy as np
import random
import entropix.core.evaluator as evaluator

__all__ = ('sample_dimensions')

logger = logging.getLogger(__name__)


def remove_dimensions(singvectors_filepath, vocab_filepath):
    """Remove top or bottom dimensions."""
    logger.info('Sampling dimensions...')
    model = np.load(singvectors_filepath)
    left_idx, right_idx, sim = evaluator.load_vocab_and_men_data(vocab_filepath)
    init_men_spr = evaluator.evaluate_on_men(model, left_idx, right_idx, sim)
    for dim_idx in range(model.shape[1]-1):
        reduced = model[:, dim_idx:]
        men_spr = evaluator.evaluate_on_men(reduced, left_idx, right_idx, sim)
        print(reduced.shape[1], men_spr)



def evaluate(model, left_idx, right_idx, sim, dataset):
    if dataset not in ['men', 'simlex', 'simverb']:
        raise Exception('Unsupported eval dataset: {}'.format(dataset))
    if dataset == 'men':
        return evaluator.evaluate_on_men(model, left_idx, right_idx, sim)


def increase_dim(model, keep, dims, left_idx, right_idx, sim, dataset,
                 output_basename, iterx, shuffle, mode, rate):
    logger.info('Increasing dimensions to maximize score. Iteration = {}'
                .format(iterx))
    max_spr = evaluate(model[:, list(keep)], left_idx, right_idx, sim, dataset)
    for idx, dim_idx in enumerate(dims):
        keep.add(dim_idx)
        spr = evaluate(model[:, list(keep)], left_idx, right_idx, sim, dataset)
        if spr > max_spr:
            max_spr = spr
            logger.info('New max = {} with dim = {} at idx = {}'
                        .format(max_spr, len(keep), dim_idx))
        else:
            keep.remove(dim_idx)
        if mode == 'mix' and idx % rate == 0:
            keep = reduce_dim(model, keep, left_idx, right_idx, sim, dataset,
                              max_spr, output_basename, iterx, step=1,
                              shuffle=shuffle, save=False)
    if shuffle:
        keep_filepath = '{}.iter-{}.shuffle.txt'.format(output_basename, iterx)
    else:
        keep_filepath = '{}.iter-{}.txt'.format(output_basename, iterx)
    logger.info('Finished dim increase. Saving list of keep idx to {}'
                .format(keep_filepath))
    with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
        print('\n'.join([str(idx) for idx in sorted(keep)]), file=keep_stream)
    return keep, max_spr


def reduce_dim(model, keep, left_idx, right_idx, sim, dataset, max_spr,
               output_basename, iterx, step, shuffle, save):
    logger.info('Reducing dimensions while maintaining highest score. '
                'Step = {}'.format(step))
    remove = set()
    if shuffle:
        dim_indexes = list(keep)
        random.shuffle(dim_indexes)
    else:
        dim_indexes = sorted(keep)
    for dim_idx in dim_indexes:
        dims = keep.difference(remove)
        dims.remove(dim_idx)
        spr = evaluate(model[:, list(dims)], left_idx, right_idx, sim, dataset)
        if spr >= max_spr:
            remove.add(dim_idx)
            logger.info('Constant max = {} removing dim_idx = {}. New dim = {}'
                        .format(max_spr, dim_idx, len(dims)))
    if shuffle:
        reduce_filepath = '{}.iter-{}.reduce.step-{}.shuffle.txt'.format(
            output_basename, iterx, step)
    else:
        reduce_filepath = '{}.iter-{}.reduce.step-{}.txt'.format(
            output_basename, iterx, step)
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
        reduce_dim(model, keep, left_idx, right_idx, sim, dataset, max_spr,
                   output_basename, iterx, step, shuffle, save)
    return keep


def sample_dimensions(singvectors_filepath, vocab_filepath, dataset,
                      output_basename, num_iter, shuffle, mode, rate):
    model = np.load(singvectors_filepath)
    logger.info('Sampling dimensions over a total of {} dims, optimizing '
                'on {} using {} mode...'
                .format(model.shape[1], dataset, mode))
    if mode not in ['seq', 'mix']:
        raise Exception('Unsupported mode {}'.format(mode))
    if shuffle:
        logger.info('Shuffling mode ON')
    else:
        logger.info('Shuffling mode OFF')
    model = model[:, ::-1]  # put singular vectors in decreasing order of singular value
    if dataset not in ['men', 'simlex', 'simverb']:
        raise Exception('Unsupported eval dataset: {}'.format(dataset))
    if dataset == 'men':
        left_idx, right_idx, sim = evaluator.load_vocab_and_men_data(vocab_filepath)
    elif dataset == 'simlex':
        pass
    elif dataset == 'simverb':
        pass
    keep = set([0, 1])  # start at 2-dims
    for iterx in range(1, num_iter+1):
        dims = [idx for idx in list(range(model.shape[1])) if idx not in keep]
        if shuffle:
            random.shuffle(dims)
        keep, max_spr = increase_dim(model, keep, dims, left_idx, right_idx,
                                     sim, dataset, output_basename, iterx,
                                     shuffle, mode, rate)
        keep = reduce_dim(model, keep, left_idx, right_idx, sim, dataset,
                          max_spr, output_basename, iterx, step=1,
                          shuffle=shuffle, save=True)