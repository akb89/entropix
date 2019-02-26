"""Generate plots and analysis from results"""

import os
import entropix

if __name__ == '__main__':
    print('Running entropix XPVIZ#001')

    OUTPUT_DIRPATH = '/home/ludovica/Desktop/entropix_analysis/'
    assert os.path.exists(OUTPUT_DIRPATH)

    MODELS_DIRPATH = '/home/ludovica/Desktop/debug_out/ppmi/'
    assert os.path.exists(OUTPUT_DIRPATH)
