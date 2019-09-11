"""Entropy-based informativeness computation.

Entropy is computed from a gensim W2V CBOW language model.
"""

from functools import lru_cache

import logging
import scipy
import numpy as np
from gensim.models import Word2Vec

__all__ = ('Informativeness')

logger = logging.getLogger(__name__)


class Informativeness():
    """Informativeness as defined in the Kabbach et al 2019 paper."""

    def __init__(self, model_path):
        """Initialize class as a gensim W2V model."""
        logger.info('Loading gensim W2V CBOW model...')
        self._model = Word2Vec.load(model_path)

    @lru_cache(maxsize=50)
    def _get_prob_distribution(self, context):
        words_and_probs = self._model.predict_output_word(
            context, topn=len(self._model.wv.vocab))
        return [item[1] for item in words_and_probs]

    @lru_cache(maxsize=50)
    def context_informativeness(self, context):
        """Get context informativeness (CI)."""
        probs = self._get_prob_distribution(context)
        shannon_entropy = scipy.stats.entropy(probs)
        ctx_ent = 1 - (shannon_entropy / np.log(len(probs)))
        return ctx_ent

    @lru_cache(maxsize=50)
    def context_word_informativeness(self, context, word_index):
        """Get context word informativeness (CWI).

        Word is specified by its index in the given context.
        """
        ctx_info_with_word = self.context_informativeness(context)
        ctx_without_word = tuple(x for idx, x in enumerate(context) if
                                 idx != word_index)
        if not ctx_without_word:
            if ctx_info_with_word > .25:  # arbitrary value set to keep relatively informative words in particular contexts
                return 1
            return -1
        ctx_info_without_word = self.context_informativeness(ctx_without_word)
        return ctx_info_with_word - ctx_info_without_word

    def filter_context_words(self, context):
        """Return only words in context with positive CWI."""
        filtered_context = []
        if not context:
            return filtered_context
        for idx, word in enumerate(context):
            if self.context_word_informativeness(context, idx) > 0:
                filtered_context.append(word)
        return filtered_context
