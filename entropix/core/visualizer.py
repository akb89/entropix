""""
Visualization tools.
"""

import logging
import collections
import math
import seaborn as sns
import matplotlib.pyplot as plt
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


def visualize_boxplot(output_dirpath, max_n, input_filepaths):
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

#        dists[lab] = d_mask
        dists.append(d)

    df = pd.DataFrame(data=dists)
#    ax = sns.heatmap(data=df)
    ax = sns.violinplot(data=dists, orient="h", palette="Set2")
    fig = ax.get_figure()
    fig.savefig(graph_outfile)
