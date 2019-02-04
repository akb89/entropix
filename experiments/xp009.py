"""Generating entropies and counts for a set of wikipedia dumps."""

import os
import functools
import multiprocessing

import entropix
import entropix.utils.files as futils


def _process(model_dirpath, min_count, win_size, wiki_filepath):
    entropix.generate(corpus_filepath=wiki_filepath,
                      output_dirpath=model_dirpath, min_count=min_count,
                      win_size=win_size)
    return wiki_filepath


if __name__ == '__main__':
    print('Running entropix XP#009')

    WIKI_DIRPATH = '/home/kabbach/witokit/data/wiki/'
    MODEL_DIRPATH = '/home/kabbach/entropix/models/'
    NUM_THREADS = 22
    MIN_COUNT = 300
    WIN_SIZE = 5

    assert os.path.exists(WIKI_DIRPATH)
    assert os.path.exists(MODEL_DIRPATH)

    FILE_NUM = 0
    WIKI_FILEPATHS = futils.get_input_filepaths(WIKI_DIRPATH)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process, MODEL_DIRPATH, MIN_COUNT,
                                    WIN_SIZE)
        for wikipath, corpus_size, vocab_size, entropy in pool.imap_unordered(process, WIKI_FILEPATHS):
            FILE_NUM += 1
            print('Done processing file {}'.format(wikipath))
            print('Completed processing of {}/{} files'
                  .format(FILE_NUM, len(WIKI_FILEPATHS)))
