"""Dimensionality reduction through dimensionality selection."""

import logging

import numpy as np

import entropix.core.evaluator as evaluator

__all__ = ('sample_dimensions')

logger = logging.getLogger(__name__)


def two_pass():
    pass


def _sample_dimensions():
    pass


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


def sample_dimensions(singvectors_filepath, vocab_filepath, dataset,
                      keep_filepath, keep_reduced_filepath):
    model = np.load(singvectors_filepath)
    logger.info('Sampling dimensions over a total of {} dims, optimizing '
                'on {}...'.format(model.shape[1], dataset))
    model = model[:, ::-1]  # put singular vectors in decreasing order of singular value
    if dataset not in ['men', 'simlex', 'simverb']:
        raise Exception('Unsupported eval dataset: {}'.format(dataset))
    if dataset == 'men':
        left_idx, right_idx, sim = evaluator.load_vocab_and_men_data(vocab_filepath)
    elif dataset == 'simlex':
        pass
    elif dataset == 'simverb':
        pass
    first_pass_dims = list(range(model.shape[1]))[2:]
    keep = [0, 1]  # start at 2-dims
    logger.info('Starting first pass: adding dimensions to maximize score')
    max_spr = evaluate(model[:, keep], left_idx, right_idx, sim, dataset)
    for dim_idx in first_pass_dims:
        keep.append(dim_idx)
        reduced = model[:, keep]
        spr = evaluate(reduced, left_idx, right_idx, sim, dataset)
        if spr > max_spr:
            max_spr = spr
            logger.info('New max = {} with dim = {} at idx = {}'
                        .format(max_spr, len(keep), dim_idx))
        else:
            keep.pop()
    logger.info('Completed first pass. Saving list of keep idx to {}'
                .format(keep_filepath))
    with open(keep_filepath, 'w', encoding='utf-8') as keep_stream:
        print(keep, file=keep_stream)
    logger.info('Starting second pass: reducing dimensions while maintaining '
                'score')
    remove = []
    for dim_idx in keep:
        dims = [idx for idx in keep if idx not in remove and idx != dim_idx]
        reduced = model[:, dims]
        spr = evaluate(reduced, left_idx, right_idx, sim, dataset)
        if spr >= max_spr:
            remove.append(dim_idx)
            logger.info('Constant max = {} removing dim_idx = {}. New dim = {}'
                        .format(max_spr, dim_idx, len(dims)))
    logger.info('Finished second pass. Saving list of reduced keep idx to {}'
                .format(keep_reduced_filepath))
    with open(keep_reduced_filepath, 'w', encoding='utf-8') as reduced_stream:
        print([idx for idx in keep if idx not in remove], file=reduced_stream)
