"""Generating singular vectors distributions"""

import os
import multiprocessing
import glob
import entropix
import entropix.utils.files as futils


def process(model_filepath):
    global OUTPUT_DIRPATH
    entropix.singvectors_dist(OUTPUT_DIRPATH, model_filepath)
    print('DONE: {}'.format(model_filepath))

if __name__ == '__main__':
    print('Running entropix XP#010')

    MODEL_DIRPATH = '/mnt/8tera/shareclic/ludovica.pannitto/models/'
    OUTPUT_DIRPATH = '/mnt/8tera/shareclic/ludovica.pannitto/entropix_output/'

    models_files = glob.glob('{}*.npz'.format(MODEL_DIRPATH))

    p = multiprocessing.Pool(20)

    p.map(process, models_files)
