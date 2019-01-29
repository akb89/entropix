"""Generating entropies and counts for a set of wikipedia dumps."""

import os
import functools
import multiprocessing

import entropix
import entropix.utils.files as futils


def _process(counts_dirpath, wiki_filepath):
    counts = entropix.count(counts_dirpath, wiki_filepath)
    return wiki_filepath, entropix.compute(counts)


if __name__ == '__main__':
    print('Running entropix XP001')

    WIKI_DIRPATH = '/Users/akb/Github/entropix/data/wiki/'
    COUNTS_DIRPATH = '/Users/akb/Github/entropix/data/counts/'
    RESULTS_FILEPATH = '/Users/akb/Github/entropix/data/xp001.results'
    NUM_THREADS = 2

    assert os.path.exists(WIKI_DIRPATH)
    assert os.path.exists(COUNTS_DIRPATH)

    file_num = 0
    results = {}
    wiki_filepaths = futils.get_input_filepaths(WIKI_DIRPATH)
    with multiprocessing.Pool(NUM_THREADS) as pool:
        process = functools.partial(_process, COUNTS_DIRPATH)
        for wikipath, corpus_size, vocab_size, entropy in pool.imap_unordered(process, wiki_filepaths):
            file_num += 1
            print('Done processing file {}'.format(wikipath))
            print('Completed processing of {}/{} files'
                  .format(file_num, len(wiki_filepaths)))
            partial = {
                'corpus_size': corpus_size,
                'vocab_size': vocab_size,
                'entropy': entropy
            }
            results[os.path.basename(wikipath)] = partial
    print('Saving results to file {}'.format(RESULTS_FILEPATH))
    with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as output_stream:
        print('{:20}\t{:5}\t{:5}\t{:5}'
              .format('Language', 'Corpus size', 'Vocab size', 'Entropy'),
              file=output_stream)
        print('-*80', file=output_stream)
        for key in sorted(results.keys()):
            print('{:20}\t{:5}\t{:5}\t{:5}'
                  .format(key, results[key]['corpus_size'],
                          results[key]['vocab_size'], results[key]['entropy']),
                  file=output_stream)
