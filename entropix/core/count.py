"""Count words."""

import os

from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

__all__ = ('count_words')


def count_words(output_dirpath, corpus_filepath, min_count=0):
    """Count words in a corpus."""
    if corpus_filepath.endswith('.txt'):
        output_filepath = os.path.join(
            output_dirpath, '{}.counts'.format(
                os.path.basename(corpus_filepath).split('.txt')[0]))
    else:
        output_filepath = os.path.join(
            output_dirpath,
            '{}.counts'.format(os.path.basename(corpus_filepath)))
    _counts = defaultdict(int)
    logger.info('Counting words in {}'.format(corpus_filepath))
    with open(corpus_filepath, 'r', encoding='utf-8') as input_stream:
        for line in input_stream:
            line = line.strip()
            for word in line.split():
                _counts[word] += 1
    if min_count == 0:
        counts = _counts
    else:
        counts = {word: count for word, count in _counts.items() if count >= min_count}
    logger.info('Saving counts to {}'.format(output_filepath))
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        for word, wcount in sorted(counts.items(),
                                   key=lambda x: (x[1], x[0]), reverse=True):
            print('{}\t{}'.format(word, wcount), file=output_stream)
    logger.info('Done counting words in {}'.format(corpus_filepath))
    return counts
