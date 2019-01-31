"""Compute entropy metrics."""

import math
import logging

logger = logging.getLogger(__name__)

__all__ = ('compute_entropy')


def compute_entropy(counts):
    """Compute entropy from a counts dict."""
    total_count = sum(counts.values())
    logger.info('Total vocab size = {}'.format(total_count))
    corpus_size = 0
    if total_count > 1e9:
        corpus_size = '{}B'.format(round(total_count / 1e9))
    elif total_count > 1e6:
        corpus_size = '{}M'.format(round(total_count / 1e6))
    elif total_count > 1e3:
        corpus_size = '{}K'.format(round(total_count / 1e3))
    else:
        corpus_size = '{}'.format(round(total_count))
    logger.info('Corpus size = {}'.format(corpus_size))
    vocab_size = 0
    if len(counts) > 1e6:
        vocab_size = '{}M'.format(round(len(counts) / 1e6))
    elif len(counts) > 1e3:
        vocab_size = '{}K'.format(round(len(counts) / 1e3))
    else:
        vocab_size = '{}'.format(round(len(counts)))
    logger.info('Vocab size = {}'.format(vocab_size))
    probs = [count/total_count for count in counts.values()]
    logprobs = [prob * math.log2(prob) for prob in probs]
    entropy = -sum(logprobs)
    logger.info('Entropy = {}'.format(entropy))
    return corpus_size, vocab_size, entropy
