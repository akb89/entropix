"""Generate a set of aligned vocabularies for each model."""
import os

from collections import defaultdict
from tqdm import tqdm

import numpy as np

import entropix.utils.data as dutils
import entropix.utils.files as futils
import entropix.utils.metrix as metrix
import entropix.core.aligner as aligner
import entropix.core.matrixor as matrixor
import entropix.core.evaluator as evaluator


def dump_vocabs(models, dirpath):
    print('Dumping aligned models and vocabularies...')
    for name1, model1, vocab1 in tqdm(models):
        aligned_model1 = model1
        vocab = vocab1
        for name2, model2, vocab2 in models:
            if name1 == name2:
                continue
            assert aligned_model1.shape[1] == model2.shape[1]
            aligned_model1, _, vocab = aligner.align_vocab(
                aligned_model1, model2, vocab, vocab2)
        vocab_path = os.path.join(dirpath, '{}-aligned.vocab'.format(name1))
        model_path = os.path.join(dirpath, '{}-aligned.npy'.format(name1))
        futils.save_vocab(vocab, vocab_path)
        np.save(model_path, aligned_model1)


def load_models(model_names, model_dirpath, start, end):
    loaded_models = []
    for model_name in model_names:
        print('Loading model {}...'.format(model_name))
        model_path = os.path.join(model_dirpath, '{}.npy'.format(model_name))
        vocab_path = os.path.join(model_dirpath, '{}.vocab'.format(model_name))
        model, vocab = dutils.load_model_and_vocab(
            model_path, 'numpy', vocab_path, start=start, end=end)
        loaded_models.append((model_name, model, vocab))
    return loaded_models


def launch_xp(model_names, model_dirpath, start, end):
    models = load_models(model_names, model_dirpath, start, end)
    dump_vocabs(models, model_dirpath)


if __name__ == '__main__':
    SVD_DIRPATH = '/home/kabbach/entropix/models/svd'
    START = 0
    END = 10000
    print('Aligning vocabularies across all models')
    MODEL_NAMES = ['enwiki07', 'oanc', 'enwiki2', 'acl', 'enwiki4', 'bnc']
    launch_xp(MODEL_NAMES, SVD_DIRPATH, START, END)
