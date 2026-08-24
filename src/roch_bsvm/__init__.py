"""Public API for the improved robust BSVM implementation."""

from .author_bsvm import AuthorExtendedBSVMClassifier, VALID_AUTHOR_VARIANTS
from .model import BinaryTreeBSVMClassifier
from .search import DEFAULT_CI_CANDIDATES, tune_ci_strategy

__all__ = [
    "AuthorExtendedBSVMClassifier",
    "BinaryTreeBSVMClassifier",
    "DEFAULT_CI_CANDIDATES",
    "VALID_AUTHOR_VARIANTS",
    "tune_ci_strategy",
]

__version__ = "0.2.0"
