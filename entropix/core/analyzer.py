""""A set of qualitative analysis on the datasets, based on the ppmi matrix."""

import logging
import numpy as np
from scipy import stats
import collections

import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('analyze_overlap')


def analyze_overlap(model_filepath, vocab_filepath, dataset_name):

    dataset_left, dataset_right, dataset_sim = dutils.load_word_pairs_and_sim(dataset_name)

    vocab_mapping = dutils.load_vocab(vocab_filepath)
    reverse_vocab_mapping = {y:x for x, y in vocab_mapping.items()}

    pairs_dataset = list(zip(dataset_left, dataset_right))
    idx_pairs_dataset = [(vocab_mapping[x], vocab_mapping[y]) for x, y in pairs_dataset]

    model = dutils.load_model_from_npz(model_filepath)


    number_of_shared_dims = {}
    top_shared_dims = {}
    bottom_shared_dims = {}
    value_shared_dims = {}


    for left_w in dataset_left:
        number_of_shared_dims[left_w] = {}
        top_shared_dims[left_w] = {}
        bottom_shared_dims[left_w] = {}
        value_shared_dims[left_w] = {}

        # x_row = model.getrow(vocab_mapping[left_w])
        #
        # for right_w in dataset_right:
        #     y_row = model.getrow(vocab_mapping[right_w])
        #
        #     x_inters_y = x_row.minimum(y_row)
        #     nonzero_indices = x_inters_y.nonzero()[1]
        #
        #     nonzero_values = [x_inters_y[0,i] for i in nonzero_indices]
        #     nonzero_indices = [reverse_vocab_mapping[el] for el in nonzero_indices]
        #
        #     number_of_shared_dims[left_w][right_w] = len(nonzero_values)
        #
        #     sorted_value_idx_pair = sorted(zip(nonzero_values,nonzero_indices), reverse = True)
        #
        #     top_shared_dims[left_w][right_w] = [el[1] for el in sorted_value_idx_pair[:10]]
        #     value_shared_dims[left_w][right_w] = sorted_value_idx_pair[9][0]


#            print(left_w, right_w)
#            print(number_of_shared_dims[left_w][right_w])
#            print(top_shared_dims[left_w][right_w])
#            print(value_shared_dims[left_w][right_w])
#            input()

    i = 0
    for left_w, right_w in pairs_dataset:

        x_row = model.getrow(vocab_mapping[left_w])
        y_row = model.getrow(vocab_mapping[right_w])

        x_inters_y = x_row.minimum(y_row)
        nonzero_indices = x_inters_y.nonzero()[1]

        nonzero_values = [x_inters_y[0,i] for i in nonzero_indices]
        nonzero_indices = [reverse_vocab_mapping[el] for el in nonzero_indices]

        number_of_shared_dims[left_w][right_w] = len(nonzero_values)

        sorted_value_idx_pair = sorted(zip(nonzero_values,nonzero_indices), reverse = True)

        top_shared_dims[left_w][right_w] = [el[1] for el in sorted_value_idx_pair[:10]]
        bottom_shared_dims[left_w][right_w] = [el[1] for el in sorted_value_idx_pair[-10:]]
        value_shared_dims[left_w][right_w] = (sorted_value_idx_pair[0][0], sorted_value_idx_pair[9][0], sorted_value_idx_pair[len(nonzero_values)//2][0], sum(el[0] for el in sorted_value_idx_pair)/len(sorted_value_idx_pair))

        print(left_w, right_w, dataset_sim[i])
        i+=1
        print('Number of shared contexts: {}'.format(number_of_shared_dims[left_w][right_w]))
        print('Top 10 shared contexts:', top_shared_dims[left_w][right_w])
        print('Bottom 10 shared contexts:', bottom_shared_dims[left_w][right_w])
        print('Top PPMI, 10-th PPMI, Median PPMI, avg PPMI:', value_shared_dims[left_w][right_w])
        print()
#        input()

#        print(x_inters_y)
#        input()


    n_shared_contexts = [number_of_shared_dims[x][y] for x, y in pairs_dataset]
    top_ppmi = [value_shared_dims[x][y][0] for x, y in pairs_dataset]
    ppmi_10 = [value_shared_dims[x][y][1] for x, y in pairs_dataset]
    median_ppmi = [value_shared_dims[x][y][2] for x, y in pairs_dataset]
    avg_ppmi = [value_shared_dims[x][y][3] for x, y in pairs_dataset]

    print ('CORRELATIONS')
    print ('Number of shared contexts', stats.spearmanr(n_shared_contexts, dataset_sim)[0])
    print ('Top PPMI of shared contexts', stats.spearmanr(top_ppmi, dataset_sim)[0])
    print ('10-th PPMI of shared contexts', stats.spearmanr(ppmi_10, dataset_sim)[0])
    print ('Median PPMI', stats.spearmanr(median_ppmi, dataset_sim)[0])
    print ('avg_ppmi', stats.spearmanr(avg_ppmi, dataset_sim)[0])
