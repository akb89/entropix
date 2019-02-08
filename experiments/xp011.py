"""Generate cosine similarities distribution for a model."""

import entropix

if __name__ == '__main__':
    print('Running entropix XP#011')

    _OUTPUT_DIRPATH = '/home/ludovica/Desktop/debug_out/'
    _MODEL_FILEPATH = '/home/ludovica/Desktop/debug_out/itwiki.20190120.mincount-100.win-2.npz'
    _NUM_THREADS = 4
    _BIN_SIZE = 0.1

    entropix.pwcosine_distribution(_OUTPUT_DIRPATH, _MODEL_FILEPATH,
                                   _NUM_THREADS, _BIN_SIZE)
