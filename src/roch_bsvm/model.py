"""Scikit-learn compatible BSVM with candidate insertion strategies."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .scoring import compute_ci_priority


class BinaryTreeBSVMClassifier(ClassifierMixin, BaseEstimator):
    """Robust binary SVM using improved priorities and candidate insertion.

    The estimator first creates the paper's feasible core by removing training
    points that are misclassified or violate the unit margin. Candidate points
    are then sorted by ``ci_strategy``. With ``insertion_strategy="binary_tree"``,
    the ordered list is split into a best half and a remaining half. Each half
    is tried as one block; a failing block is recursively bisected until it is
    accepted or reduced to a rejected singleton. With
    ``insertion_strategy="sequential"``, candidates are tried one by one, which
    is useful as an original-algorithm baseline.

    This class intentionally handles binary classification. For multiclass
    data, wrap it in ``sklearn.multiclass.OneVsRestClassifier``.
    """

    def __init__(
        self,
        *,
        kernel: str = "rbf",
        C: float = 10.0,
        subproblem_C: float = 10_000.0,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        class_weight: str | Mapping[Any, float] | None = "balanced",
        ci_strategy: str = "robust_hybrid",
        ci_params: Mapping[str, Any] | None = None,
        initial_margin: float = 1.0,
        feasibility_margin: float = 0.0,
        feasibility_tolerance: float = 1e-7,
        max_support_fraction: float = 1.0,
        insertion_strategy: str = "binary_tree",
        shrinking: bool = True,
        cache_size: float = 200.0,
        random_state: int | None = 42,
        verbose: int = 0,
    ) -> None:
        self.kernel = kernel
        self.C = C
        self.subproblem_C = subproblem_C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.class_weight = class_weight
        self.ci_strategy = ci_strategy
        self.ci_params = ci_params
        self.initial_margin = initial_margin
        self.feasibility_margin = feasibility_margin
        self.feasibility_tolerance = feasibility_tolerance
        self.max_support_fraction = max_support_fraction
        self.insertion_strategy = insertion_strategy
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.random_state = random_state
        self.verbose = verbose

    def _make_svc(self, C: float) -> SVC:
        return SVC(
            C=C,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            class_weight=None,
            shrinking=self.shrinking,
            cache_size=self.cache_size,
            random_state=self.random_state,
        )

    def _resolve_class_weights(
        self, original_y: np.ndarray, encoded_y: np.ndarray
    ) -> dict[int, float]:
        labels, counts = np.unique(encoded_y, return_counts=True)
        if self.class_weight is None:
            return {int(label): 1.0 for label in labels}
        if self.class_weight == "balanced":
            total = len(encoded_y)
            return {
                int(label): total / (len(labels) * int(count))
                for label, count in zip(labels, counts, strict=True)
            }
        if not isinstance(self.class_weight, Mapping):
            raise ValueError("class_weight must be None, 'balanced', or a mapping")
        resolved: dict[int, float] = {}
        for encoded_label, original_label in zip((-1, 1), self.classes_, strict=True):
            if original_label not in self.class_weight:
                raise ValueError(f"class_weight is missing label {original_label!r}")
            resolved[encoded_label] = float(self.class_weight[original_label])
        return resolved

    def _ensure_two_classes_in_core(
        self, core_mask: np.ndarray, encoded_y: np.ndarray, signed_margin: np.ndarray
    ) -> np.ndarray:
        core_mask = core_mask.copy()
        for label in (-1, 1):
            if np.any(core_mask & (encoded_y == label)):
                continue
            label_indices = np.flatnonzero(encoded_y == label)
            best = label_indices[int(np.argmax(signed_margin[label_indices]))]
            core_mask[best] = True
        return core_mask

    def _feasibility(
        self, model: SVC, X: np.ndarray, y: np.ndarray
    ) -> tuple[bool, float, float]:
        signed_margin = y * np.asarray(model.decision_function(X)).reshape(-1)
        minimum = float(np.min(signed_margin))
        required = float(self.feasibility_margin) - float(self.feasibility_tolerance)
        support_fraction = float(len(model.support_) / len(X))
        feasible = minimum > required and support_fraction <= float(
            self.max_support_fraction
        )
        return feasible, minimum, support_fraction

    def _fit_trial(
        self, X: np.ndarray, y: np.ndarray, indices: list[int]
    ) -> tuple[SVC, bool, float, float]:
        model = self._make_svc(float(self.subproblem_C))
        model.fit(X[indices], y[indices])
        self.n_model_fits_ += 1
        feasible, minimum, support_fraction = self._feasibility(
            model, X[indices], y[indices]
        )
        return model, feasible, minimum, support_fraction

    def fit(self, X: Any, y: Any) -> "BinaryTreeBSVMClassifier":
        """Fit the robust binary-tree BSVM."""

        started = time.perf_counter()
        X_checked, y_checked = check_X_y(X, y, dtype=float, ensure_min_samples=2)
        self.n_features_in_ = X_checked.shape[1]

        self.label_encoder_ = LabelEncoder().fit(y_checked)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "BinaryTreeBSVMClassifier requires exactly two classes; "
                "use OneVsRestClassifier for multiclass data"
            )
        encoded_zero_one = self.label_encoder_.transform(y_checked)
        encoded_y = np.where(encoded_zero_one == 0, -1, 1).astype(int)
        self.class_weight_map_ = self._resolve_class_weights(y_checked, encoded_y)

        if float(self.C) <= 0 or float(self.subproblem_C) <= 0:
            raise ValueError("C and subproblem_C must be positive")
        if not 0 < float(self.max_support_fraction) <= 1:
            raise ValueError("max_support_fraction must be in (0, 1]")
        if self.insertion_strategy not in {"binary_tree", "sequential"}:
            raise ValueError(
                "insertion_strategy must be 'binary_tree' or 'sequential'"
            )

        initial_model = self._make_svc(float(self.C))
        initial_model.fit(X_checked, encoded_y)
        initial_decision = np.asarray(initial_model.decision_function(X_checked)).reshape(-1)
        initial_prediction = initial_model.predict(X_checked)
        initial_signed_margin = encoded_y * initial_decision
        core_mask = (initial_prediction == encoded_y) & (
            initial_signed_margin >= float(self.initial_margin) - 1e-9
        )
        core_mask = self._ensure_two_classes_in_core(
            core_mask, encoded_y, initial_signed_margin
        )

        accepted = np.flatnonzero(core_mask).astype(int).tolist()
        candidates = np.flatnonzero(~core_mask).astype(int)
        rejected: list[int] = []
        self.initial_core_indices_ = np.asarray(accepted, dtype=int)
        self.initial_candidate_indices_ = candidates.copy()
        self.history_: list[dict[str, Any]] = []
        self.n_batch_attempts_ = 0
        self.n_model_fits_ = 0

        current_model, feasible, minimum, support_fraction = self._fit_trial(
            X_checked, encoded_y, accepted
        )
        if not feasible:
            raise RuntimeError(
                "The initial feasible core could not be separated by the chosen "
                "kernel. Try a larger subproblem_C or a smaller initial_margin."
            )

        self.initial_core_min_margin_ = minimum
        self.initial_core_support_fraction_ = support_fraction

        if len(candidates):
            candidate_decisions = np.asarray(
                current_model.decision_function(X_checked[candidates])
            ).reshape(-1)
            ci_result = compute_ci_priority(
                strategy=self.ci_strategy,
                X=X_checked,
                y=encoded_y,
                candidate_indices=candidates,
                decision_values=candidate_decisions,
                class_weights=self.class_weight_map_,
                params=self.ci_params,
            )
            order = np.argsort(-ci_result.scores, kind="stable")
            ordered_candidates = candidates[order].astype(int).tolist()
            self.initial_ci_scores_ = {
                int(index): float(score)
                for index, score in zip(candidates, ci_result.scores, strict=True)
            }
            self.initial_ci_components_ = {
                name: {
                    int(index): float(value)
                    for index, value in zip(candidates, values, strict=True)
                }
                for name, values in ci_result.components.items()
            }

            def try_block(block: list[int], depth: int) -> None:
                nonlocal current_model
                if not block:
                    return
                self.n_batch_attempts_ += 1
                trial_indices = accepted + block
                trial, is_feasible, min_margin, sv_fraction = self._fit_trial(
                    X_checked, encoded_y, trial_indices
                )
                event: dict[str, Any] = {
                    "attempt": self.n_batch_attempts_,
                    "depth": depth,
                    "batch_size": len(block),
                    "candidate_indices": list(block),
                    "min_signed_margin": min_margin,
                    "support_fraction": sv_fraction,
                }
                if is_feasible:
                    accepted.extend(block)
                    current_model = trial
                    event["action"] = "accepted"
                    self.history_.append(event)
                    if self.verbose:
                        print(
                            f"[accept] depth={depth} size={len(block)} "
                            f"min_margin={min_margin:.4g}"
                        )
                    return

                if len(block) == 1:
                    rejected.append(block[0])
                    event["action"] = "rejected"
                    self.history_.append(event)
                    if self.verbose:
                        print(
                            f"[reject] depth={depth} index={block[0]} "
                            f"min_margin={min_margin:.4g}"
                        )
                    return

                event["action"] = "split"
                self.history_.append(event)
                midpoint = math.ceil(len(block) / 2)
                try_block(block[:midpoint], depth + 1)
                try_block(block[midpoint:], depth + 1)

            if self.insertion_strategy == "sequential":
                for candidate in ordered_candidates:
                    try_block([candidate], depth=0)
            else:
                # The root itself is not tested: the requested procedure starts
                # by trying the best half, then the remaining half.
                if len(ordered_candidates) == 1:
                    try_block(ordered_candidates, depth=0)
                else:
                    root_midpoint = math.ceil(len(ordered_candidates) / 2)
                    try_block(ordered_candidates[:root_midpoint], depth=1)
                    try_block(ordered_candidates[root_midpoint:], depth=1)
        else:
            self.initial_ci_scores_ = {}
            self.initial_ci_components_ = {}

        self.model_ = current_model
        self.selected_indices_ = np.asarray(sorted(accepted), dtype=int)
        self.rejected_indices_ = np.asarray(sorted(rejected), dtype=int)
        self.n_support_vectors_ = int(len(self.model_.support_))
        self.fit_time_seconds_ = float(time.perf_counter() - started)
        self.fit_summary_ = {
            "n_samples": int(len(X_checked)),
            "initial_core": int(len(self.initial_core_indices_)),
            "initial_candidates": int(len(self.initial_candidate_indices_)),
            "accepted_candidates": int(
                len(self.selected_indices_) - len(self.initial_core_indices_)
            ),
            "rejected_candidates": int(len(self.rejected_indices_)),
            "selected_total": int(len(self.selected_indices_)),
            "support_vectors": self.n_support_vectors_,
            "batch_attempts": int(self.n_batch_attempts_),
            "model_fits": int(self.n_model_fits_),
            "fit_time_seconds": self.fit_time_seconds_,
            "ci_strategy": self.ci_strategy,
            "insertion_strategy": self.insertion_strategy,
        }
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        X_checked = check_array(X, dtype=float)
        return np.asarray(self.model_.decision_function(X_checked))

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        encoded = np.asarray(self.model_.predict(check_array(X, dtype=float)))
        zero_one = np.where(encoded == -1, 0, 1)
        return self.label_encoder_.inverse_transform(zero_one)
