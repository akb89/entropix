"""Compute raw, singvalues entropy and 90%Energy dimension on PPMI models.

To generate a set of PPMI models from a set of RAW models, use xp113.py
"""

import os
import multiprocessing

import numpy as np
import scipy

import entropix.utils.files as futils


def _process(raw_model_filepath):
    print('Computing rank-related metrics on raw model {}'
          .format(raw_model_filepath))
    model = np.load(raw_model_filepath)
    rank = len(model)
    entropy = scipy.stats.entropy(model, base=2)
    energy = np.sum(model**2)
    reduced_energy = .9 * energy
    reduced_rank = rank
    for idx, _ in enumerate(model):
        if np.sum(model[idx:]**2) < reduced_energy:
            break
        reduced_rank = rank - idx
    results = {'rank': rank, 'hsigma': entropy, 'red_e_rank': reduced_rank}
    return os.path.basename(raw_model_filepath), results


if __name__ == '__main__':
    XP_NUM = '013'
    print('Running entropix XP#{}'.format(XP_NUM))

    PPMI_MODELS_DIRPATH = '/home/kabbach/entropix/models/ppmi/'
    RESULTS_DIRPATH = '/home/kabbach/entropix/results/'
    NUM_THREADS = 51

    assert os.path.exists(PPMI_MODELS_DIRPATH)
    assert os.path.exists(RESULTS_DIRPATH)

    ppmi_results_filepath = os.path.join(RESULTS_DIRPATH, 'xp013.ppmi.results')

    file_num = 0
    results = {}
    ppmi_model_singvalues_filepaths = futils.get_singvalues_filepaths(PPMI_MODELS_DIRPATH)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        for model_basename, raw_results in pool.imap_unordered(_process, ppmi_model_singvalues_filepaths):
            file_num += 1
            print('Done processing model {}'.format(model_basename))
            print('Completed processing of {}/{} files'
                  .format(file_num, len(ppmi_model_singvalues_filepaths)))
            results[model_basename] = raw_results
    with open(ppmi_results_filepath, 'w', encoding='utf-8') as raw_stream:
        print('{:47}\t{:>5}\t{:>20}\t{:>5}'
              .format('Model', 'Rank', 'H(Sigma)', '0.9E-Rank'),
              file=raw_stream)
        for key in sorted(results):
            print('{:47}\t{:>5}\t{:>20}\t{:>5}'
                  .format(key, results[key]['rank'], results[key]['hsigma'],
                          results[key]['red_e_rank']), file=raw_stream)