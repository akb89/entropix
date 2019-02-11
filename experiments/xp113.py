"""Generate PPMI models from RAW models."""
import os

import entropix
import entropix.utils.files as futils


if __name__ == '__main__':
    XP_NUM = '113'
    print('Running entropix XP#{}'.format(XP_NUM))

    RAW_MODEL_DIRPATH = '/home/kabbach/entropix/models/raw/'
    PPMI_MODEL_DIRPATH = '/home/kabbach/entropix/models/ppmi/'
    COUNTS_DIRPATH = '/home/kabbach/witokit/data/counts/xp001/'

    assert os.path.exists(RAW_MODEL_DIRPATH)
    assert os.path.exists(PPMI_MODEL_DIRPATH)

    file_num = 0
    raw_model_filepaths = futils.get_models_filepaths(RAW_MODEL_DIRPATH)
    for raw_model_filepath in raw_model_filepaths:
        print('Processing file {}'.format(raw_model_filepath))
        file_num += 1
        model_basename = os.path.basename(raw_model_filepath).split('.mincount')[0]
        counts_filepath = os.path.join(COUNTS_DIRPATH,
                                       '{}.counts'.format(model_basename))
        entropix.weigh(PPMI_MODEL_DIRPATH, raw_model_filepath, counts_filepath,
                       'ppmi')
        print('Done processing model {}'.format(model_basename))
        print('Completed processing of {}/{} files'
              .format(file_num, len(raw_model_filepaths)))
