"""Dimensionality reduction through dimensionality selection."""

import logging
import random
import numpy as np

import entropix.core.evaluator as evaluator

__all__ = ('sample_dimensions')

logger = logging.getLogger(__name__)


def increase_dim(model, dataset, keep, dims, left_idx, right_idx, sim,
                 output_basename, iterx, shuffle, mode, rate):
    logger.info('Increasing dimensions to maximize score. Iteration = {}'
                .format(iterx))
    max_spr = evaluator.evaluate(model[:, list(keep)], left_idx, right_idx, sim, dataset)
    init_len_keep = len(keep)
    added_counter = 0
    for idx, dim_idx in enumerate(dims):
        keep.add(dim_idx)
        spr = evaluator.evaluate(model[:, list(keep)], left_idx, right_idx, sim, dataset)
        if spr > max_spr:
            added_counter += 1
            max_spr = spr
            logger.info('New max = {} with dim = {} at idx = {}'
                        .format(max_spr, len(keep), dim_idx))
            if mode == 'mix' and added_counter % rate == 0:
                keep = reduce_dim(model, keep, left_idx, right_idx, sim,
                                  max_spr, output_basename, iterx, step=1,
                                  shuffle=shuffle, save=False)
        else:
            keep.remove(dim_idx)

    if shuffle:
        keep_filepath = '{}.keep.shuffled.iter-{}.txt'.format(output_basename, iterx)
    else:
        keep_filepath = '{}.keep.iter-{}.txt'.format(output_basename, iterx)
    logger.info('Finished dim increase. Saving list of keep idx to {}'
                .format(keep_filepath))
    with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
        print('\n'.join([str(idx) for idx in sorted(keep)]), file=keep_stream)
    return keep, max_spr


def reduce_dim(model, dataset, keep, left_idx, right_idx, sim, max_spr,
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
        spr = evaluator.evaluate(model[:, list(dims)], left_idx, right_idx, sim, dataset)
        if spr >= max_spr:
            remove.add(dim_idx)
            logger.info('Constant max = {} removing dim_idx = {}. New dim = {}'
                        .format(max_spr, dim_idx, len(dims)))
    if shuffle:
        reduce_filepath = '{}.keep.shuffled.iter-{}.reduce.step-{}.txt'.format(
            output_basename, iterx, step)
    else:
        reduce_filepath = '{}.keep.iter-{}.reduce.step-{}.txt'.format(
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
        reduce_dim(model, dataset, keep, left_idx, right_idx, sim, max_spr,
                   output_basename, iterx, step, shuffle, save)
    return keep


def sample_seq_mix(model, dataset, left_idx, right_idx, sim, output_basename,
                   num_iter, shuffle, mode, rate, start, end):
    if shuffle:
        logger.info('Shuffling mode ON')
    else:
        logger.info('Shuffling mode OFF')
    logger.info('Iterating over {} dims starting at {} and ending at {}'
                .format(model.shape[1], start, end))
    keep = set([start, start+1])  # start at 2-dims
    for iterx in range(1, num_iter+1):
        dims = [idx for idx in list(range(model.shape[1]))[start:end] if idx not in keep]
        if shuffle:
            random.shuffle(dims)
        keep, max_spr = increase_dim(model, dataset, keep, dims, left_idx,
                                     right_idx, sim, output_basename, iterx,
                                     shuffle, mode, rate)
        keep = reduce_dim(model, dataset, keep, left_idx, right_idx, sim,
                          max_spr, output_basename, iterx, step=1,
                          shuffle=shuffle, save=True)


def sample_limit(model, dataset, left_idx, right_idx, sim, output_basename, limit,
                 start, end, rewind):
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
            spr = evaluator.evaluate(model[:, list(dims)], left_idx, right_idx, sim, dataset)
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
                    spr = evaluator.evaluate(model[:, list(dims)], left_idx, right_idx, sim, dataset)
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


def sample_dimensions(singvectors_filepath, vocab_filepath, dataset,
                      output_basename, num_iter, shuffle, mode, rate,
                      start, end, limit, rewind):
    model = np.load(singvectors_filepath)
    logger.info('Sampling dimensions over a total of {} dims, optimizing '
                'on {} using {} mode...'
                .format(model.shape[1], dataset, mode))
    if mode not in ['seq', 'mix', 'limit']:
        raise Exception('Unsupported mode {}'.format(mode))
    model = model[:, ::-1]  # put singular vectors in decreasing order of singular value
    if end > model.shape[1]:
        raise Exception('End parameter is > model.shape[1]: {} > {}'
                        .format(end, model.shape[1]))
    if end == 0:
        end = model.shape[1]
    if dataset not in ['men', 'simlex', 'simverb', 'sts2012']:
        raise Exception('Unsupported eval dataset: {}'.format(dataset))
    left_idx, right_idx, sim = evaluator.load_words_and_sim_(vocab_filepath,
                                                             dataset)
    if mode in ['seq', 'mix']:
        sample_seq_mix(model, dataset, left_idx, right_idx, sim,
                       output_basename, num_iter, shuffle, mode, rate, start,
                       end)
    if mode == 'limit':
        sample_limit(model, dataset, left_idx, right_idx, sim, output_basename,
                     limit, start, end, rewind)
