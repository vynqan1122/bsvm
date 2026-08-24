"""Validation-based selection of a c_i priority function."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score

from .model import BinaryTreeBSVMClassifier


DEFAULT_CI_CANDIDATES: tuple[dict[str, Any], ...] = (
    {"ci_strategy": "paper_priority", "ci_params": {"eps": 1e-6}},
    {"ci_strategy": "boundary_exp", "ci_params": {"class_power": 1.0}},
    {
        "ci_strategy": "robust_hybrid",
        "ci_params": {
            "n_neighbors": 5,
            "class_power": 1.0,
            "boundary_power": 1.0,
            "local_power": 1.0,
            "density_power": 0.5,
        },
    },
    {
        "ci_strategy": "robust_hybrid",
        "ci_params": {
            "n_neighbors": 9,
            "class_power": 1.5,
            "boundary_power": 1.0,
            "local_power": 1.5,
            "density_power": 1.0,
        },
    },
)


def _validation_score(
    y_train: np.ndarray, y_valid: np.ndarray, y_pred: np.ndarray, scoring: str
) -> float:
    if scoring == "balanced_accuracy":
        return float(balanced_accuracy_score(y_valid, y_pred))
    if scoring != "minority_f1":
        raise ValueError("scoring must be 'minority_f1' or 'balanced_accuracy'")
    labels, counts = np.unique(y_train, return_counts=True)
    minority_label = labels[int(np.argmin(counts))]
    return float(f1_score(y_valid, y_pred, pos_label=minority_label, zero_division=0))


def tune_ci_strategy(
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
    *,
    base_estimator: BinaryTreeBSVMClassifier | None = None,
    candidates: Iterable[Mapping[str, Any]] = DEFAULT_CI_CANDIDATES,
    scoring: str = "minority_f1",
    refit: bool = True,
) -> tuple[BinaryTreeBSVMClassifier, list[dict[str, Any]]]:
    """Choose a c_i function on held-out validation data.

    The function returns the best estimator and a JSON-friendly result table.
    When ``refit=True``, the best configuration is trained again on the union
    of training and validation data.
    """

    base = base_estimator or BinaryTreeBSVMClassifier()
    y_train_array = np.asarray(y_train)
    y_valid_array = np.asarray(y_valid)
    results: list[dict[str, Any]] = []
    best_score = -np.inf
    best_config: dict[str, Any] | None = None

    for raw_config in candidates:
        config = dict(raw_config)
        estimator = clone(base).set_params(**config)
        estimator.fit(X_train, y_train)
        prediction = estimator.predict(X_valid)
        score = _validation_score(
            y_train_array, y_valid_array, prediction, scoring
        )
        row = {
            "ci_strategy": config["ci_strategy"],
            "ci_params": dict(config.get("ci_params") or {}),
            "validation_score": score,
            "support_vectors": estimator.n_support_vectors_,
            "accepted_candidates": estimator.fit_summary_["accepted_candidates"],
            "rejected_candidates": estimator.fit_summary_["rejected_candidates"],
            "batch_attempts": estimator.n_batch_attempts_,
        }
        results.append(row)
        if score > best_score:
            best_score = score
            best_config = config

    if best_config is None:
        raise ValueError("candidates must contain at least one configuration")

    best_estimator = clone(base).set_params(**best_config)
    if refit:
        X_combined = np.concatenate([np.asarray(X_train), np.asarray(X_valid)], axis=0)
        y_combined = np.concatenate([y_train_array, y_valid_array], axis=0)
        best_estimator.fit(X_combined, y_combined)
    else:
        best_estimator.fit(X_train, y_train)

    results.sort(key=lambda row: row["validation_score"], reverse=True)
    return best_estimator, results

