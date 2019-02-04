"""Expose functions outside module."""

from .core.counter import count_words as count
from .core.calculator import compute_entropy as compute
from .core.evaluator import evaluate_distributional_space as evaluate
from .core.generator import generate_distributional_model as generate
from .core.reducer import reduce_matrix_via_svd as reduce
from .utils.files import get_sparsematrix_filepath as get_model_filepath
from .utils.files import get_sing_vectors_filepath as get_sing_vectors_filepath
from .utils.files import get_sing_values_filepath as get_sing_values_filepath

__all__ = ('count', 'compute', 'evaluate', 'generate', 'reduce',
           'get_model_filepath', 'get_sing_vectors_filepath',
           'get_sing_values_filepath')
