"""Measure nearest neighbor overlap between two numpy models."""
import logging
import multiprocessing
import functools
import numpy as np
import scipy.spatial as spatial
from tqdm import tqdm
import entropix.utils.metrix as metrix

logger = logging.getLogger(__name__)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def _get_n_nearest_neighbors(idx, model, n):
    vector = model[idx]
    sim = []
    for i in range(model.shape[0]):
        if i == idx:
            sim.append(0)
        else:
            ivector = model[i]
            cos = metrix.cosine_similarity(vector, ivector)
            sim.append(cos)
    return set(np.argsort(sim)[::-1][:n])


def _get_variance(model1, model2, n, idx):
    neighb1 = _get_n_nearest_neighbors(idx, model1, n)
    neighb2 = _get_n_nearest_neighbors(idx, model2, n)
    return 1 - len(neighb1.intersection(neighb2))/n


def _process(model1, model2, n, batchidx):
    return [_get_variance(model1, model2, n, idx) for idx in tqdm(batchidx)]


def _compare_low_ram(model1, model2, n, num_threads=1):
    variance = []
    assert model1.shape[0] == model2.shape[0]
    n_chunks = round(model1.shape[0] / num_threads)
    batchidx = chunks(range(model1.shape[0]), n_chunks)
    with multiprocessing.Pool(num_threads) as pool:
        process = functools.partial(_process, model1, model2, n)
        for _var in pool.imap_unordered(process, batchidx):
            variance.extend(_var)
    return variance


def _compare_fast(model1, model2, n):
    # compute cosine similarities
    logger.info('Computing cosine similarities...')
    cos1 = 1 - spatial.distance.cdist(model1, model1, 'cosine')
    cos2 = 1 - spatial.distance.cdist(model2, model2, 'cosine')
    # postprocess to avoid taking the word itself as its own nearest neighbor
    cos1 = cos1 - np.diag(np.ones(model1.shape[0]))
    cos2 = cos2 - np.diag(np.ones(model2.shape[0]))
    # compute the n-nearest neighbor indices
    logger.info('Computing nearest neighbor indices...')
    idx1 = np.argpartition(cos1, -n, axis=1)[:, -n:]
    idx2 = np.argpartition(cos2, -n, axis=1)[:, -n:]
    # sort to be able to use np.count_nonzero(A==B, axis=1)
    idx1.sort()
    idx2.sort()
    logger.info('Computing variance...')
    # compute the variance following Pierrejean and Tanguy 2018
    return 1 - np.count_nonzero(idx1 == idx2, axis=1) / n


def _compare(model1, model2, num_neighbors, num_threads, low_ram):
    if low_ram:
        logger.warning('Forcing low RAM comparison. This should be slower, '
                       'but you can pass in a --num-threads parameter to '
                       'speed up the process')
        return _compare_low_ram(model1, model2, num_neighbors, num_threads)
    try:
        return _compare_fast(model1, model2, num_neighbors)
    except MemoryError:
        logger.warning('Models are too big to fit in RAM. Switching to low '
                       'RAM footprint algorithm. You can pass in a '
                       '--num-threads parameter to speed up the process')
        return _compare_low_ram(model1, model2, num_neighbors, num_threads)


def compare(model1, model2, vocab1, vocab2, num_neighbors, num_threads,
            low_ram):
    model1, model2 = align_vocab(model1, model2, vocab1, vocab2)
    variance = _compare(model1, model2, num_neighbors, num_threads, low_ram)
    # take the average and std
    avg = np.mean(variance)
    std = np.std(variance)
    return avg, std
