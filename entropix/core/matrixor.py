"""Matrix transformations.

Implements algorithm 2.4 (AO+Scaling) of the paper:

@article{devetal2018,
    title={{Absolute Orientation for Word Embedding Alignment}},
    author={Sunipa Dev and Safia Hassan and Jeff M. Phillips},
    journal={CoRR},
    year={2018},
    volume={abs/1806.01330}
}
"""

import numpy as np

from einsumt import einsumt

__all__ = ('apply_absolute_orientation_with_scaling')


def center(matrix):
    """Center input matrix."""
    return matrix - matrix.mean(axis=0)


def compute_sum_inner_product(A, B):
    # return np.diagonal(A @ B.T).sum()
    #return np.diagonal(np.einsum('ij,ik->jk', A, B)).sum()
    return np.diagonal(einsumt('ij,ik->jk', A, B)).sum()


def compute_scaling(A, B):
    return compute_sum_inner_product(A, B) / (np.linalg.norm(B, ord='fro') ** 2)


def compute_sum_outer_product(A, B):
    # return B.T @ A
    #return np.einsum('ij,ik->jk', B, A)  # much faster than above
    return einsumt('ij,ik->jk', B, A)


def apply_ao_rotation(A, B):
    """Apply algo 2.1: SVD-based rotation."""
    H = compute_sum_outer_product(A, B)
    U, _, VT = np.linalg.svd(H)  # decompose
    R = U.dot(VT)  # build rotation
    return B.dot(R)


def apply_absolute_orientation_with_scaling(A, B):
    """Apply algo 2.4."""
    BR = apply_ao_rotation(A, B)  # rotated B
    s = compute_scaling(A, BR)
    return s * BR
