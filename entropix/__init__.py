"""Expose functions outside module."""

from .main import _sample as sample
from .core.evaluator import evaluate as evaluate_on_splits
