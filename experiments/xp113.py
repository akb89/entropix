"""Generate PPMI models from RAW models."""
import os
import functools
import multiprocessing

import entropix
import entropix.utils.files as futils


def _process(ppmi_output_dirpath, counts_dirpath, raw_model_filepath):
    model_basename = os.path.basename(raw_model_filepath).split('.mincount')[0]
    counts_filepath = os.path.join(counts_dirpath,
                                   '{}.counts'.format(model_basename))
    entropix.weigh(ppmi_output_dirpath, raw_model_filepath, counts_filepath,
                   'ppmi')
    return raw_model_filepath


if __name__ == '__main__':
    XP_NUM = '113'
    print('Running entropix XP#{}'.format(XP_NUM))

    RAW_MODEL_DIRPATH = '/home/kabbach/entropix/models/raw/'
    PPMI_MODEL_DIRPATH = '/home/kabbach/entropix/models/ppmi/'
    COUNTS_DIRPATH = '/home/kabbach/witokit/data/counts/xp001/'
    NUM_THREADS = 51

    assert os.path.exists(RAW_MODEL_DIRPATH)
    assert os.path.exists(PPMI_MODEL_DIRPATH)

    file_num = 0
    raw_model_filepaths = futils.get_models_filepaths(RAW_MODEL_DIRPATH)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process, PPMI_MODEL_DIRPATH,
                                    COUNTS_DIRPATH)
        for model_basename in pool.imap_unordered(process, raw_model_filepaths):
            file_num += 1
            print('Done processing model {}'.format(model_basename))
            print('Completed processing of {}/{} files'
                  .format(file_num, len(raw_model_filepaths)))
