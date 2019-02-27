"""Generate plots and analysis from results"""

import os
import glob
import entropix
import entropix.core.extractor as extractor

if __name__ == '__main__':
    print('Running entropix XPVIZ#001')

    OUTPUT_DIRPATH = '/home/ludovica/Desktop/entropix_analysis/'
    assert os.path.exists(OUTPUT_DIRPATH)

    MODELS_DIRPATH = '/home/ludovica/Desktop/debug_out/ppmi/'
    assert os.path.exists(MODELS_DIRPATH)

    vocab_filepath = '/home/ludovica/Downloads/enwiki.20190120.mincount-300.win-2.vocab'

    # Issue 1:
    # Relevant words for each dimension. We're taking the 10 highest positive
    # values and the 10 lowest negative values

    for sing_vectors_filepath in os.listdir(MODELS_DIRPATH):
        sing_vectors_filepath = os.path.join(MODELS_DIRPATH, sing_vectors_filepath)
        extractor.extract_top_participants(
            OUTPUT_DIRPATH, sing_vectors_filepath, vocab_filepath, 20, True)


    # Issue 2:
    # How much do dimensions semantically overlap?

    for filename in glob.glob(OUTPUT_DIRPATH+'*.CM')
