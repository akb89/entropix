""""
Visualization tools.
"""

import logging
import collections
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('visualize_heatmap', 'visualize_singvalues')


def visualize_heatmap(output_dirpath, input_filpath, filter_filepath):

    heatmap_outfile = futils.get_png_filename(output_dirpath, input_filpath)

    filter = None
    if filter_filepath:
        filter = dutils.load_index_set(filter_filepath)

    matrix = dutils.load_2d_array(input_filpath, filter, symm=True)

    mask = np.zeros_like(matrix)
    mask[np.tril_indices_from(mask)] = 1
    with sns.axes_style("white"):
        ax = sns.heatmap(matrix, mask=mask, square=True, cmap="Greens")

    fig = ax.get_figure()
    fig.savefig(heatmap_outfile)


def visualize_singvalues(output_dirpath, input_filepath, filter_filepath):
    graph_outfile = futils.get_png_filename(output_dirpath, input_filepath)

    filter = None
    if filter_filepath:
        filter = dutils.load_index_set(filter_filepath)

    sing_values = np.load(input_filepath)
    if filter:
        sing_values = [sing_values[i] for i in sorted(filter)]

    ax = sns.lineplot(x=range(len(sing_values)), y=sing_values)
    fig = ax.get_figure()
    fig.savefig(graph_outfile)


def visualize_ipr_scatter(output_dirpath, input_filepath, filter_filepath):
    graph_outfile = futils.get_png_filename(output_dirpath, input_filepath)

    filter = None
    if filter_filepath:
        filter = dutils.load_index_set(filter_filepath)

    x, y = dutils.load_2columns(input_filepath)
    if filter:
        x = [x[i] for i in sorted(filter)]
        y = [y[i] for i in sorted(filter)]

    ax = sns.scatterplot(x=x, y=y)
    fig = ax.get_figure()
    fig.savefig(graph_outfile)


def visualize_boxplot_per_dataset(output_dirpath, max_n, input_filepaths):
    graph_outfile = futils.get_png_filename(output_dirpath, input_filepaths[0])
#    dists = {}
    dists = []
    for filename in input_filepaths:
        # if 'shuffle' in filename:
        #     lab = 'shuffle'
        # else:
        #     lab = 'linear'
        if 'men' in filename:
            lab = 'men'
        elif 'simlex' in filename:
            lab ='simlex'
        else:
            lab = 'simverb'
        d = dutils.load_intlist(filename)
        d_mask = [0]*max_n
        for i in d:
            d_mask[i]=1

#        dists[lab] = d
        dists.append(d)

#    df = pd.DataFrame(data=dists)
#    ax = sns.heatmap(data=df)
#    ax = sns.boxplot(data=dists, orient="h", palette="Set2")
    ax = sns.swarmplot(data=dists, orient="h", palette="Set2")
    fig = ax.get_figure()
    fig.savefig(graph_outfile)

def visualize_boxplot_per_shuffle(output_dirpath, max_n, input_filepaths):
    graph_outfile = futils.get_png_filename(output_dirpath, input_filepaths[0])
#    dists = {}
    dists = []
    for filename in input_filepaths:
        if 'shuffle' in filename:
            lab = 'shuffle'
        else:
            lab = 'linear'

        d = dutils.load_intlist(filename)
        d_mask = [0]*max_n
        for i in d:
            d_mask[i]=1

#        dists[lab] = d
        dists.append(d)


#    ax = sns.heatmap(data=df)
#    ax = sns.boxplot(data=dists, orient="h", palette="Set2")
    ax = sns.swarmplot(data=dists, orient="h", palette="Set2")
    fig = ax.get_figure()
    fig.savefig(graph_outfile)


def visualize_heatmap_per_shuffle(output_dirpath, max_n, input_filepaths):
    graph_outfile = futils.get_png_filename(output_dirpath, input_filepaths[0])
#    dists = {}
    dists = []
    for filename in input_filepaths:
        if 'shuffle' in filename:
            lab = 'shuffle'
        else:
            lab = 'linear'

        d = dutils.load_intlist(filename)
        print(d)
        d_mask = [0]*max_n
        for i in d:
            d_mask[i]=1

#        dists[lab] = d_mask
        dists.append(d_mask)

#    df = pd.DataFrame(data=dists)
    ax = sns.heatmap(data=dists)
#    ax = sns.boxplot(data=dists, orient="h", palette="Set2")
    fig = ax.get_figure()
    fig.savefig(graph_outfile)


def _compute_mean_distance(idxs_list, model):
    vectors = [model[x, :] for x in idxs_list]
    centroid = np.sum(vectors, axis=0)

    d = 0
    for vector in vectors:
        cos = 1-cosine(vector, centroid)
        d+=cos/len(vectors)

    return d

def visualize_cosine_similarity(output_dirpath, wordlist_pos_filepath,
                                 wordlist_neg_filepath, model, vocabulary):

    word_to_idx = {y:x for x, y in dutils.load_vocab_mapping(vocabulary).items()}
    model = np.load(model)

    wordlist_pos = dutils.load_wordlist(wordlist_pos_filepath)
    wordlist_neg = dutils.load_wordlist(wordlist_neg_filepath)

    positive_cosines = []
    negative_cosines = []
    cosines = []
    x_dims = []
    for idx in wordlist_pos:
        x_dims.append(idx)
        wpos = wordlist_pos[idx]
        wneg = wordlist_neg[idx]

        wpos_idxs = [word_to_idx[x] for x in wpos]
        wneg_idxs = [word_to_idx[x] for x in wneg]

        mean_distance_pos = _compute_mean_distance(wpos_idxs, model)
        mean_distance_neg = _compute_mean_distance(wneg_idxs, model)


        cosines.append(max(mean_distance_pos, mean_distance_neg))

        positive_cosines.append(mean_distance_pos)
        negative_cosines.append(mean_distance_neg)

    positive_graph_outfile = futils.get_png_filename(output_dirpath, '{}.cosines'.format(wordlist_pos_filepath))
    print(x_dims)
    ax = sns.scatterplot(x=x_dims, y=cosines)
    fig = ax.get_figure()
    fig.savefig(positive_graph_outfile)

#    negative_graph_outfile = futils.get_png_filename(output_dirpath, 'negative_cosines')
#    ax2 = sns.lineplot(x=range(len(positive_cosines)), y=negative_cosines)
#    fig2 = ax2.get_figure()
#    fig2.savefig(negative_graph_outfile)
#    print(positive_cosines)
#    print(negative_cosines)
