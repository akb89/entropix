"""Generating entropies and counts for a set of wikipedia dumps."""

import os
import functools
import multiprocessing

import entropix
import entropix.utils.files as futils


def _process(counts_dirpath, wiki_filepath):
    counts = entropix.count(counts_dirpath, wiki_filepath)
    corpus_size, vocab_size, entropy = entropix.compute(counts)
    return wiki_filepath, corpus_size, vocab_size, entropy


if __name__ == '__main__':
    print('Running entropix XP#001')

    WIKI_DIRPATH = '/home/kabbach/witokit/data/wiki/'
    COUNTS_DIRPATH = '/home/kabbach/entropix/data/counts/xp001/'
    RESULTS_FILEPATH = '/home/kabbach/entropix/xp001.results'
    NUM_THREADS = 51

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
                'entropy': round(entropy, 2)
            }
            results[os.path.basename(wikipath)] = partial
    print('Saving results to file {}'.format(RESULTS_FILEPATH))
    with open(RESULTS_FILEPATH, 'w', encoding='utf-8') as output_stream:
        print('{:20}\t{:>11}\t{:>10}\t{:>7}'
              .format('Wiki', 'Corpus size', 'Vocab size', 'Entropy'),
              file=output_stream)
        print('-'*63, file=output_stream)
        for key in sorted(results.keys()):
            print('{:20}\t{:>11}\t{:>10}\t{:>7}'
                  .format(key, results[key]['corpus_size'],
                          results[key]['vocab_size'], results[key]['entropy']),
                  file=output_stream)
