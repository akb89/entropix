""""
Extract words with largest absolute values in singular vectors.
"""

import logging
import collections
import numpy as np

import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ()


def extract_top_participants(output_dirpath, sing_vectors_filepath,
                             vocab_filepath, num_top_elements):

    vocab_mapping = dutils.load_vocab_mapping(vocab_filepath)
    sing_vectors = np.load(sing_vectors_filepath)

    output_filepath = futils.get_topelements_filepath(output_dirpath,
                                                      sing_vectors_filepath,
                                                      num_top_elements)
    ret = []
    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        for i, column in enumerate(sing_vectors.T):
            sortedcol = sorted(range(len(column)),
                               key=lambda x: abs(column[x]),
                               reverse=True)[:num_top_elements]
            ret.append(sortedcol)
            words = [vocab_mapping[j] for j in sortedcol]
            print(i, ', '.join(words[:num_top_elements]), file=output_stream)
    return ret
