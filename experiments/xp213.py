"""SVD-reduce a set of ppmi count models generated after xp113."""

import os
import entropix
import entropix.utils.files as futils

if __name__ == '__main__':
    print('Running entropix XP#213')

    MODEL_DIRPATH = '/home/kabbach/entropix/models/ppmi/'

    assert os.path.exists(MODEL_DIRPATH)

    FILE_NUM = 0
    MODEL_FILEPATHS = futils.get_models_filepaths(MODEL_DIRPATH)
    FILE_NUM = 0
    for model_filepath in MODEL_FILEPATHS:
        FILE_NUM += 1
        print('Applying SVD to {}'.format(model_filepath))
        sing_values_filepath = entropix.get_singvalues_filepath(model_filepath)
        sing_vectors_filepath = entropix.get_singvectors_filepath(model_filepath)
        for dim in [100, 300, 500, 1000, 2000, 5000, 10000, 0]:
            try:
                print('Reducing matrix via SVD and k = {}'.format(dim))
                U, S = entropix.svd(
                    model_filepath=model_filepath, dim=dim,
                    sing_values_filepath=sing_values_filepath,
                    sing_vectors_filepath=sing_vectors_filepath, compact=True)
                print('Final matrix rank = {}'.format(S.size))
                break
            except Exception as err:
                print('All singular values are non-null. Trying again')
                continue
        print('Done reducing {}/{} models'
              .format(FILE_NUM, len(MODEL_FILEPATHS)))
