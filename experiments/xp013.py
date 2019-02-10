"""Compute raw, singvalues entropy and 90%Energy dimension on PPMI models.

To generate a set of PPMI models from a set of RAW models, use xp113.py
"""

import os
import functools
import multiprocessing

import entropix
import entropix.utils.files as futils


def _process(raw_model_filepath):
    print('Computing rank-related metrics on raw model {}'
          .format(raw_model_filepath))
    raw = {}
    ppmi = {}

    return os.path.basename(raw_model_filepath), raw, ppmi


if __name__ == '__main__':
    XP_NUM = '012'
    print('Running entropix XP#{}'.format(XP_NUM))

    RAW_MODELS_DIRPATH = '/home/kabbach/entropix/models/raw/'
    PPMI_MODELS_DIRPATH = '/home/kabbach/entropix/models/ppmi/'
    RESULTS_DIRPATH = '/home/kabbach/entropix/results/'
    NUM_THREADS = 22

    assert os.path.exists(RAW_MODELS_DIRPATH)
    assert os.path.exists(RESULTS_DIRPATH)

    raw_results_filpath = os.path.join(RESULTS_DIRPATH, 'xp012.raw.results')
    ppmi_results_filepath = os.path.join(RESULTS_DIRPATH, 'xp012.ppmi.results')

    file_num = 0
    results = {'raw': {}, 'ppmi': {}}
    raw_model_filepaths = futils.get_raw_model_filepaths(RAW_MODELS_DIRPATH)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        #process = functools.partial(_process, COUNTS_DIRPATH, MIN_COUNT)
        for model_basename, raw, ppmi in pool.imap_unordered(_process, raw_model_filepaths):
            file_num += 1
            print('Done processing model {}'.format(model_basename))
            print('Completed processing of {}/{} files'
                  .format(file_num, len(raw_model_filepaths)))
            results['raw'][model_basename] = raw
            results['ppmi'][model_basename] = ppmi
    with open(raw_results_filpath, 'w', encoding='utf-8') as raw_stream:
        print('{:20}\t{:>5}\t{:>10}\t{:>10}'
              .format('LANG', 'Rank', 'H(Sigma)', '0.9E-Rank'),
              file=raw_stream)
        for key, value in results['raw'].items():
            print('{}\t{}\t{}\t{}'
                  .format(key, value['rank'], value['hsigma'],
                          value['red_e_rank']), file=raw_stream)
    with open(ppmi_results_filepath, 'w', encoding='utf-8') as ppmi_stream:
        print('{:20}\t{:>5}\t{:>10}\t{:>10}'
              .format('LANG', 'Rank', 'H(Sigma)', '0.9E-Rank'),
              file=ppmi_stream)
        for key, value in results['ppmi'].items():
            print('{}\t{}\t{}\t{}'
                  .format(key, value['rank'], value['hsigma'],
                          value['red_e_rank']), file=ppmi_stream)
    # print('Saving results to file {}'.format(RESULTS_FILEPATH))
    # with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as output_stream:
    #     print('{:20}\t{:>11}\t{:>10}\t{:>7}'
    #           .format('Wiki', 'Corpus size', 'Vocab size', 'Entropy'),
    #           file=output_stream)
    #     print('-'*63, file=output_stream)
    #     for key in sorted(results.keys()):
    #         print('{:20}\t{:>11}\t{:>10}\t{:>7}'
    #               .format(key, results[key]['corpus_size'],
    #                       results[key]['vocab_size'], results[key]['entropy']),
    #               file=output_stream)
