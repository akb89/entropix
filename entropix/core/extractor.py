""""
Extract words with largest absolute values in singular vectors.
"""

import logging
import collections
import numpy as np

import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('extract_top_participants')


def _compute_intersection(lists, output_filepath):

    with open(output_filepath, 'w', encoding='utf-8') as output_stream:
        print('\t'.join(str(x) for x in range(len(lists))), file=output_stream)
        for idx1, l1 in enumerate(lists):
            s = '{}\t'.format(idx1)
            for i in range(idx1):
                s += '\t'
            for idx2, l2 in enumerate(lists[idx1+1:]):
                intersection = list(set(l1) & set(l2))
                s += '{}\t'.format(len(intersection))
            print(s, file=output_stream)




def extract_top_participants(output_dirpath, sing_vectors_filepath,
                             vocab_filepath, num_top_elements, save_cm):

    vocab_mapping = dutils.load_vocab_mapping(vocab_filepath)
    sing_vectors = np.load(sing_vectors_filepath)

    output_filepath_abs = futils.get_topelements_filepath(output_dirpath,
                                                          sing_vectors_filepath,
                                                          num_top_elements, 'abs')
    output_filepath_pos = futils.get_topelements_filepath(output_dirpath,
                                                          sing_vectors_filepath,
                                                          num_top_elements, 'pos')
    output_filepath_neg = futils.get_topelements_filepath(output_dirpath,
                                                          sing_vectors_filepath,
                                                          num_top_elements, 'neg')
    output_filepath_abscm = futils.get_topelements_filepath(output_dirpath,
                                                            sing_vectors_filepath,
                                                            num_top_elements, 'abs.CM')
    output_filepath_poscm = futils.get_topelements_filepath(output_dirpath,
                                                            sing_vectors_filepath,
                                                            num_top_elements, 'pos.CM')
    output_filepath_negcm = futils.get_topelements_filepath(output_dirpath,
                                                            sing_vectors_filepath,
                                                            num_top_elements, 'neg.CM')
    ret_abs, ret_pos, ret_neg = [], [], []
    with open(output_filepath_abs, 'w', encoding='utf-8') as output_stream_abs,\
      open(output_filepath_pos, 'w', encoding='utf-8') as output_stream_pos,\
      open(output_filepath_neg, 'w', encoding='utf-8') as output_stream_neg:
        for i, column in enumerate(sing_vectors.T):

            column = list(zip(range(len(column)), column))

            abs_column = [(x, abs(y)) for x, y in column]

            sorted_col = sorted(column, key=lambda x: x[1],
                                reverse=True)
            sorted_poscol = sorted_col[:num_top_elements]
            sorted_negcol = list(reversed(sorted_col[-num_top_elements:]))

            sorted_abscol = sorted(abs_column, key=lambda x: x[1],
                                   reverse=True)[:num_top_elements]

            ret_abs.append([x for x, y in sorted_abscol])
            ret_pos.append([x for x, y in sorted_poscol])
            ret_neg.append([x for x, y in sorted_negcol])
            words_abs = [vocab_mapping[j] for j, k in sorted_abscol]
            words_pos = [vocab_mapping[j] for j, k in sorted_poscol]
            words_neg = [vocab_mapping[j] for j, k in sorted_negcol]
            print('{}\t{}'.format(i, ', '.join(words_abs)),
                  file=output_stream_abs)
            print('{}\t{}'.format(i, ', '.join(words_pos)),
                  file=output_stream_pos)
            print('{}\t{}'.format(i, ', '.join(words_neg)),
                  file=output_stream_neg)

    if save_cm:
        _compute_intersection(ret_abs, output_filepath_abscm)
        _compute_intersection(ret_pos, output_filepath_poscm)
        _compute_intersection(ret_neg, output_filepath_negcm)

    return ret_abs, ret_pos, ret_neg
