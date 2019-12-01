"""Align models with different vocabularies."""
import logging
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ('_align_model_vocab')


def _align_model_vocab(model1, model2, vocab1, vocab2):
    logger.info('Aligning models and vocabularies...')
    vocab = {word: idx for word, idx in vocab1.items() if word in vocab2}
    vocab2_to_vocab1 = {idx: vocab1[word] for word, idx in vocab2.items()
                        if word in vocab1}
    assert len(vocab) == len(vocab2_to_vocab1)
    _model1 = np.empty(shape=(len(vocab2_to_vocab1), model1.shape[1]))
    idx2 = [idx for word, idx in vocab2.items() if word in vocab1]
    assert len(idx2) == len(vocab2_to_vocab1)
    _model2 = model2[idx2, :]
    for idx, item in enumerate(sorted(idx2)):
        _model1[idx] = model1[vocab2_to_vocab1[item]]
    return _model1, _model2, vocab


def align_vocab(model1, model2, vocab1, vocab2):
    """Aligning models with potentially different vocabularies."""
    if len(vocab1) != len(vocab2):
        return _align_model_vocab(model1, model2, vocab1, vocab2)
    for word, idx in vocab1.items():
        if word not in vocab2:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
        if vocab2[word] != idx:
            return _align_model_vocab(model1, model2, vocab1, vocab2)
    logger.info('Processing already aligned vocabularies')
    return model1, model2, vocab1
