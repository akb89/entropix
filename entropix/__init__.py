"""Expose functions outside module."""

from .core.count import count_words as count
from .core.compute import compute_entropy as compute
from .core.evaluate import evaluate_distributional_space as evaluate
from .core.generate import generate_distributional_model as generate

__all__ = ('count', 'compute', 'evaluate', 'generate')
