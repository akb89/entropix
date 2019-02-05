"""Generating entropies and counts for a set of wikipedia dumps."""

import os
import entropix
import entropix.utils.files as futils

if __name__ == '__main__':
    print('Running entropix XP#009')

    WIKI_DIRPATH = '/home/kabbach/witokit/data/wiki/small/'
    MODEL_DIRPATH = '/home/kabbach/entropix/models/'
    MIN_COUNT = 300
    WIN_SIZE = 5

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
        sing_vectors_filepath = entropix.get_sing_vectors_filepath(model_filepath)
        for dim in [100, 300, 500, 1000, 2000, 5000]:
            try:
                print('Reducing matrix via SVD and k = {}'.format(dim))
                U, S = entropix.reduce(
                    model_filepath=model_filepath, dim=dim,
                    sing_values_filepath=sing_values_filepath,
                    sing_vectors_filepath=sing_vectors_filepath, compact=True)
                print('Final matrix rank = {}'.format(S.size))
                break
            except Exception as err:
                print('All singular values are non-null. Trying again')
                continue
        print('Done reducing {}/{} models'.format(FILE_NUM, len(WIKI_FILEPATHS)))
