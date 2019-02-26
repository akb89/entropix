"""Semantic Textual Similarity Evaluation"""

import os
import entropix
import entropix.utils.files as futils
import numpy as np



def _evaluate(dataset, model_filepath, vocabulary_filepath, selected_dims=None):
    model = np.load(model_filepath)
    model = model[:, ::-1]  # put singular vectors in decreasing order of singular value
    if selected_dims:
        dims = []
        with open(selected_dims, 'r', encoding='utf-8') as dims_stream:
            for line in dims_stream:
                dims.append(int(line.strip()))
        model = model[:, dims]
    return entropix.core.evaluator.evaluate_distributional_space(model, vocabulary_filepath, dataset)

if __name__=='__main__':
    print('Running entropix XP#400')

    MODEL_DIRPATH = '/home/ludovica/entropix_models/'
    VOCAB_FILEPATH =  '/home/ludovica/Downloads/enwiki.20190120.mincount-300.win-5.vocab'

    assert os.path.exists(MODEL_DIRPATH)

    for model_filepath in os.listdir(MODEL_DIRPATH):
        print(model_filepath)
        print(_evaluate('sts2012', os.path.join(MODEL_DIRPATH, model_filepath), VOCAB_FILEPATH))
