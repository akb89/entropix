"""Expose functions outside module."""

from .core.counter import count_words as count
from .core.calculator import compute_entropy as compute
from .core.calculator import compute_singvectors_distribution as singvectors_dist
from .core.evaluator import evaluate_distributional_space as evaluate
from .core.generator import generate_distributional_model as generate
from .core.reducer import reduce_matrix_via_svd as reduce
from .utils.files import get_sparsematrix_filepath as get_model_filepath
from .utils.files import get_singvectors_filepath as get_singvectors_filepath
from .utils.files import get_singvalues_filepath as get_singvalues_filepath

__all__ = ('count', 'compute', 'evaluate', 'generate', 'reduce',
           'get_model_filepath', 'get_singvectors_filepath',
           'get_singvalues_filepath', 'singvectors_dist')
