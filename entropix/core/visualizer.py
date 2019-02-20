""""
Visualization tools.
"""

import logging
import collections
import seaborn as sns
import numpy as np

import entropix.utils.files as futils
import entropix.utils.data as dutils


logger = logging.getLogger(__name__)

__all__ = ('visualize_heatmap')


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
