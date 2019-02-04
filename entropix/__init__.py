"""Expose functions outside module."""

from .core.counter import count_words as count
from .core.calculator import compute_entropy as compute
from .core.evaluator import evaluate_distributional_space as evaluate
from .core.generator import generate_distributional_model as generate

__all__ = ('count', 'compute', 'evaluate', 'generate')
