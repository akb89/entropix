"""Count words."""

from collections import defaultdict
import logging

import entropix.utils.files as futils

logger = logging.getLogger(__name__)

__all__ = ('count_words')


def count_words(corpus_filepath, min_count, output_dirpath=None):
    """Count words in a corpus."""
    output_filepath = futils.get_counts_filepath(corpus_filepath,
                                                 output_dirpath)
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
        counts = {word: count for word, count in _counts.items()
                  if count >= min_count}
        logger.info('Filtering out vocabulary words with counts lower '
                    'than {}, shrinking size by {:.2f}% from {} to {}.'
                    .format(min_count, 100-len(counts)*100.0/len(_counts),
                            len(_counts), len(counts)))
    if output_dirpath:
        logger.info('Saving counts to {}'.format(output_filepath))
        with open(output_filepath, 'w', encoding='utf-8') as output_stream:
            for word, wcount in sorted(counts.items(),
                                       key=lambda x: (x[1], x[0]),
                                       reverse=True):
                print('{}\t{}'.format(word, wcount), file=output_stream)
    logger.info('Done counting words in {}'.format(corpus_filepath))
    return counts