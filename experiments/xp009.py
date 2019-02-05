"""Generating entropies and counts for a set of wikipedia dumps."""

import os

import entropix
import entropix.utils.files as futils

if __name__ == '__main__':
    print('Running entropix XP#009')

    WIKI_DIRPATH = '/home/kabbach/witokit/data/wiki/'
    MODEL_DIRPATH = '/home/kabbach/entropix/models/'
    MIN_COUNT = 300
    WIN_SIZE = 5
    DIM = 0  # apply max SVD

    assert os.path.exists(WIKI_DIRPATH)
    assert os.path.exists(MODEL_DIRPATH)

    FILE_NUM = 0
    WIKI_FILEPATHS = futils.get_input_filepaths(WIKI_DIRPATH)
    FILE_NUM = 0
    for wikipath in WIKI_FILEPATHS:
        print('Processing file {}'.format(wikipath))
        FILE_NUM += 1
        model_filepath = entropix.get_model_filepath(
            MODEL_DIRPATH, wikipath, MIN_COUNT, WIN_SIZE)
        entropix.generate(corpus_filepath=wikipath,
                          output_dirpath=MODEL_DIRPATH, min_count=MIN_COUNT,
                          win_size=WIN_SIZE)
        print('Done generating model {}'.format(model_filepath))
        print('Applying SVD to {}'.format(model_filepath))
        sing_values_filepath = entropix.get_sing_values_filepath(model_filepath)
        sing_vectors_filepaths = entropix.get_sing_vectors_filepaths(model_filepath)
        entropix.reduce(model_filepath=wikipath, dim=DIM,
                        sing_values_filepath=sing_values_filepath,
                        sing_vectors_filepaths=sing_vectors_filepaths)
        print('Done reducing {}/{} models'.format(FILE_NUM, len(WIKI_FILEPATHS)))
