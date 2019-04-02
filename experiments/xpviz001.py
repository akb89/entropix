"""Generate plots and analysis from results"""

import os
import glob
import entropix
from datetime import datetime
import entropix.core.extractor as extractor
import entropix.core.visualizer as visualizer

if __name__ == '__main__':
    print('Running entropix XPVIZ#001')

    OUTPUT_DIRPATH = '/home/ludovica/Desktop/entropix_analysis/'
    assert os.path.exists(OUTPUT_DIRPATH)

    MODELS_DIRPATH = '/home/ludovica/entropix_models/'
    assert os.path.exists(MODELS_DIRPATH)

    vocab_filepath = '/home/ludovica/Downloads/enwiki.20190120.mincount-300.win-2.vocab'

    # Issue 1:
    # Relevant words for each dimension.

    print('Issue1')

#    for sing_vectors_filepath in glob.glob(MODELS_DIRPATH+"*.k5000.*"):
#        print(sing_vectors_filepath)
#        sing_vectors_filepath = os.path.basename(sing_vectors_filepath)
#        sing_vectors_filepath = os.path.join(MODELS_DIRPATH, sing_vectors_filepath)
#    extractor.extract_top_participants(
#            OUTPUT_DIRPATH, '/home/ludovica/entropix_models/enwiki.20190120.mincount-300.win-2.ppmi.k10000.singvectors.npy', vocab_filepath, 50, True)

    # Issue 2:
    # General overlap between features

    print('Issue2')
    input_filepath = 'enwiki.20190120.mincount-300.win-2.ppmi.k10000.top-50.abs.CM'
    visualizer.visualize_heatmap(OUTPUT_DIRPATH, OUTPUT_DIRPATH+input_filepath, None)
