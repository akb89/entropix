"""Representing sampled dim as barcode."""
import os

import matplotlib.pyplot as plt

import common as com_xp

if __name__ == '__main__':
    MODELS_PATH = '/Users/akb/Gitlab/frontiers/data/'
    barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')
    fig = plt.figure()
    WIKI_MEN_DIMS = com_xp.load_dims(os.path.join(MODELS_PATH, 'enwiki2-men-seq.dims'))
    WIKI_SIMLEX_DIMS = com_xp.load_dims(os.path.join(MODELS_PATH, 'enwiki2-simlex-seq.dims'))
    ACL_MEN_DIMS = com_xp.load_dims(os.path.join(MODELS_PATH, 'acl-men-seq.dims'))
    ACL_SIMLEX_DIMS = com_xp.load_dims(os.path.join(MODELS_PATH, 'acl-simlex-seq.dims'))
    wiki_men = com_xp.load_dims_truth_array(WIKI_MEN_DIMS, start=0, end=10000)
    wiki_simlex = com_xp.load_dims_truth_array(WIKI_SIMLEX_DIMS, start=0, end=10000)
    acl_men = com_xp.load_dims_truth_array(ACL_MEN_DIMS, start=0, end=10000)
    acl_simlex = com_xp.load_dims_truth_array(ACL_SIMLEX_DIMS, start=0, end=10000)
    wiki_men_ax = fig.add_axes([0.3, 0.46, 0.6, 0.1])
    wiki_simlex_ax = fig.add_axes([0.3, 0.34, 0.6, 0.1])
    acl_men_ax = fig.add_axes([0.3, 0.22, 0.6, 0.1])
    acl_simlex_ax = fig.add_axes([0.3, 0.1, 0.6, 0.1])
    wiki_men_ax.imshow(wiki_men.reshape((1, -1)), **barprops)
    wiki_simlex_ax.imshow(wiki_simlex.reshape((1, -1)), **barprops)
    acl_men_ax.imshow(acl_men.reshape((1, -1)), **barprops)
    acl_simlex_ax.imshow(acl_simlex.reshape((1, -1)), **barprops)
    wiki_men_ax.set_axis_off()
    wiki_simlex_ax.set_axis_off()
    acl_men_ax.set_axis_off()
    acl_simlex_ax.set_axis_off()
    plt.figtext(0.13, 0.5, 'wiki2-men')
    plt.figtext(0.13, 0.38, 'wiki2-simlex')
    plt.figtext(0.13, 0.26, 'acl-men')
    plt.figtext(0.13, 0.14, 'acl-simlex')
    plt.show()
