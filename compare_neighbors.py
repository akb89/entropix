"""Compare nearest neighbors overlap across models following (Pierrejean and Tanguy, 2018)."""

import entropix.core.comparator as comparator
import entropix.utils.data as dutils


if __name__ == '__main__':
    n = 5
    MODEL1_FILEPATH = '/Users/akb/Github/entropix/models/all/men-both-alpha0-win2-kfold1-5.npy'
    MODEL2_FILEPATH = '/Users/akb/Github/entropix/models/all/men-both-alpha0-win2-kfold1-5.npy'
    VOCAB1_FILEPATH = '{}.vocab'.format(MODEL1_FILEPATH.split('.npy')[0])
    VOCAB2_FILEPATH = '{}.vocab'.format(MODEL2_FILEPATH.split('.npy')[0])
    model1, vocab1 = dutils.load_model_and_vocab(
        model_filepath=MODEL1_FILEPATH, model_type='numpy',
        vocab_filepath=VOCAB1_FILEPATH)
    model2, vocab2 = dutils.load_model_and_vocab(
        model_filepath=MODEL2_FILEPATH, model_type='numpy',
        vocab_filepath=VOCAB2_FILEPATH)
    comparator.compare(model1, model2, vocab1, vocab2, n)
