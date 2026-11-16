"""From-scratch Support Vector Machine package."""

from .kernels import (
    kernel_matrix,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    resolve_gamma,
)
from .features import add_target_encoding, prepare_engineered_frames
from .model import SVM
from .smo import KernelProvider, SMOResult, solve_smo

__all__ = [
    "SVM",
    "prepare_engineered_frames",
    "add_target_encoding",
    "KernelProvider",
    "SMOResult",
    "solve_smo",
    "kernel_matrix",
    "linear_kernel",
    "polynomial_kernel",
    "rbf_kernel",
    "resolve_gamma",
]
