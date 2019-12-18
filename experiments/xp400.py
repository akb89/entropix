"""Representing sampled dim as barcode."""
import os

import matplotlib.pyplot as plt

import common as com_xp

if __name__ == '__main__':
    MODELS_PATH = '/Users/akb/Gitlab/frontiers/data/'
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    DATASET = 'men'
    barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')
    fig = plt.figure()
    for MODEL_NAME in MODEL_NAMES:
        DIMS_FILEPATH = os.path.join(
            MODELS_PATH, '{}-{}.dims'.format(MODEL_NAME, DATASET))
        dims = com_xp.load_dims(DIMS_FILEPATH)
        x = com_xp.load_dims_truth_array(dims, start=0, end=10000)
        ax = fig.add_axes([0.3, 0.4, 0.6, 0.2])
        ax.set_axis_off()
        ax.imshow(x.reshape((1, -1)), **barprops)
        break
    plt.show()
