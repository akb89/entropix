"""Generate cosine similarities distribution for a model."""

import os
import entropix

if __name__ == '__main__':
    print('Running entropix XP#011')

    _OUTPUT_DIRPATH = '/home/ludovica/Desktop/debug_out/'
    _MODEL_FILEPATH = '/home/ludovica/Desktop/debug_out/itwiki.20190120.mincount-100.win-2.npz'
    _VOCAB_FILEPATH = '/home/ludovica/Desktop/debug_out/itwiki.20190120.mincount-100.win-2.vocab'
    _NUM_THREADS = 4
    _BIN_SIZE = 0.1

    assert os.path.exists(_OUTPUT_DIRPATH)
    assert os.path.exists(_MODEL_FILEPATH)
    assert os.path.exists(_VOCAB_FILEPATH)

    entropix.pwcosine_distribution(_OUTPUT_DIRPATH, _MODEL_FILEPATH,
                                   _VOCAB_FILEPATH, _NUM_THREADS, _BIN_SIZE)
