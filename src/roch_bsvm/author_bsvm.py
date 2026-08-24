"""Runnable extension of the BSVM code published by Mohasel and Koosha.

The upstream repository exposes ``Initialsolution``, ``extend_samples`` and
``masterproblem`` as four kernel-specific visualization scripts.  This module
keeps that control flow, turns it into a scikit-learn estimator, and makes the
candidate-priority function and insertion policy configurable.

Upstream: https://github.com/MojtabaMohasel/BSVM.git
Pinned commit used for the comparison package:
ee5a7ae7ade4977b41d604dd989b31f4e356d342
"""

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


VALID_AUTHOR_VARIANTS = (
    "author_original",
    "robust_hybrid",
    "user_formula_1",
    "user_formula_2",
)


class AuthorExtendedBSVMClassifier(ClassifierMixin, BaseEstimator):
    """Author-code BSVM with selectable priorities and insertion strategies.

    ``author_original`` uses the priority shown in Algorithm 4 and adds one
    candidate at a time.  The three new variants use binary-tree insertion.
    Priority values are recomputed after the decision boundary changes, as in
    the original ``masterproblem`` / ``extend_samples`` loop.
    """

    def __init__(
        self,
        *,
        kernel: str = "rbf",
        C: float = 10.0,
        subproblem_C: float | None = None,
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        class_weight: str | Mapping[Any, float] | None = "balanced",
        priority_strategy: str = "robust_hybrid",
        priority_params: Mapping[str, Any] | None = None,
        insertion_strategy: str = "binary_tree",
        initial_margin: float = 1.0,
        candidate_margin: float = 1.0,
        feasibility_margin: float = 0.0,
        feasibility_tolerance: float = 1e-7,
        repair_initial_core: bool = True,
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
        self.priority_strategy = priority_strategy
        self.priority_params = priority_params
        self.insertion_strategy = insertion_strategy
        self.initial_margin = initial_margin
        self.candidate_margin = candidate_margin
        self.feasibility_margin = feasibility_margin
        self.feasibility_tolerance = feasibility_tolerance
        self.repair_initial_core = repair_initial_core
        self.shrinking = shrinking
        self.cache_size = cache_size
        self.random_state = random_state
        self.verbose = verbose

    def _make_svc(self, C: float) -> SVC:
        # The author's subproblem resets class_weight to equal weights.  Class
        # imbalance enters through alpha_y in c_i, not through SVC slack costs.
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

    @property
    def _trial_C(self) -> float:
        return float(self.C if self.subproblem_C is None else self.subproblem_C)

    def _fit_trial(
        self, X: np.ndarray, y: np.ndarray, indices: list[int]
    ) -> tuple[SVC | None, bool, float]:
        model = self._make_svc(self._trial_C)
        self.n_model_fits_ += 1
        try:
            model.fit(X[indices], y[indices])
        except ValueError:
            return None, False, float("-inf")
        prediction = np.asarray(model.predict(X[indices])).reshape(-1)
        signed_margin = y[indices] * np.asarray(
            model.decision_function(X[indices])
        ).reshape(-1)
        minimum = float(np.min(signed_margin))
        required = float(self.feasibility_margin) - float(
            self.feasibility_tolerance
        )
        feasible = bool(np.all(prediction == y[indices]) and minimum > required)
        return model, feasible, minimum

    @staticmethod
    def _ensure_two_classes(
        core_mask: np.ndarray, y: np.ndarray, signed_margin: np.ndarray
    ) -> np.ndarray:
        repaired = core_mask.copy()
        for label in (-1, 1):
            if np.any(repaired & (y == label)):
                continue
            label_indices = np.flatnonzero(y == label)
            best = label_indices[int(np.argmax(signed_margin[label_indices]))]
            repaired[best] = True
        return repaired

    def _repair_core(
        self,
        X: np.ndarray,
        y: np.ndarray,
        core_indices: list[int],
        initial_signed_margin: np.ndarray,
    ) -> tuple[list[int], SVC, float]:
        """Deterministically shrink an infeasible initial core.

        The paper assumes Algorithm 2 yields a separable two-class core.  Some
        finite-C/non-PSD kernel combinations violate that assumption.  This
        repair only removes low-margin core points and records the event; it
        never reads validation or test labels.
        """

        working = list(core_indices)
        while len(working) > 2:
            model, feasible, minimum = self._fit_trial(X, y, working)
            if feasible and model is not None:
                return working, model, minimum
            counts = {label: int(np.sum(y[working] == label)) for label in (-1, 1)}
            removable = [
                index for index in working if counts[int(y[index])] > 1
            ]
            if not removable:
                break
            remove_index = min(
                removable,
                key=lambda index: (float(initial_signed_margin[index]), index),
            )
            working.remove(remove_index)

        model, feasible, minimum = self._fit_trial(X, y, working)
        if feasible and model is not None:
            return working, model, minimum

        # If the top pair is incompatible with a sigmoid/poly kernel, search a
        # small deterministic set of alternative cross-class pairs.
        per_class = {
            label: sorted(
                np.flatnonzero(y == label).astype(int).tolist(),
                key=lambda index: (-float(initial_signed_margin[index]), index),
            )[:20]
            for label in (-1, 1)
        }
        for negative_index in per_class[-1]:
            for positive_index in per_class[1]:
                pair = [negative_index, positive_index]
                model, feasible, minimum = self._fit_trial(X, y, pair)
                if feasible and model is not None:
                    return pair, model, minimum
        raise RuntimeError(
            "No separable two-class initial core was found for the chosen "
            "kernel and hyperparameters."
        )

    def _candidate_priority(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidate_indices: list[int],
        decision_values: np.ndarray,
    ):
        scoring_strategy = (
            "paper_priority"
            if self.priority_strategy == "author_original"
            else self.priority_strategy
        )
        return compute_ci_priority(
            strategy=scoring_strategy,
            X=X,
            y=y,
            candidate_indices=np.asarray(candidate_indices, dtype=int),
            decision_values=decision_values,
            class_weights=self.class_weight_map_,
            params=self.priority_params,
            validation_X=self.validation_X_,
            validation_y=self.validation_y_encoded_,
        )

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        X_validation: Any | None = None,
        y_validation: Any | None = None,
    ) -> "AuthorExtendedBSVMClassifier":
        started = time.perf_counter()
        X_checked, y_checked = check_X_y(X, y, dtype=float, ensure_min_samples=2)
        self.n_features_in_ = X_checked.shape[1]
        self.label_encoder_ = LabelEncoder().fit(y_checked)
        self.classes_ = self.label_encoder_.classes_
        if len(self.classes_) != 2:
            raise ValueError(
                "AuthorExtendedBSVMClassifier requires exactly two classes; "
                "use OneVsRestClassifier for multiclass data"
            )
        encoded_zero_one = self.label_encoder_.transform(y_checked)
        encoded_y = np.where(encoded_zero_one == 0, -1, 1).astype(int)
        self.class_weight_map_ = self._resolve_class_weights(y_checked, encoded_y)

        self.validation_X_ = None
        self.validation_y_encoded_ = None
        if X_validation is not None or y_validation is not None:
            if X_validation is None or y_validation is None:
                raise ValueError(
                    "X_validation and y_validation must be supplied together"
                )
            validation_X = check_array(X_validation, dtype=float)
            if validation_X.shape[1] != X_checked.shape[1]:
                raise ValueError(
                    "X_validation must have the same number of features as X"
                )
            validation_y = np.asarray(y_validation)
            if len(validation_X) != len(validation_y):
                raise ValueError(
                    "X_validation and y_validation must have equal length"
                )
            validation_zero_one = self.label_encoder_.transform(validation_y)
            self.validation_X_ = validation_X
            self.validation_y_encoded_ = np.where(
                validation_zero_one == 0, -1, 1
            ).astype(int)
        if self.priority_strategy == "user_formula_2" and self.validation_X_ is None:
            raise ValueError(
                "user_formula_2 requires held-out X_validation/y_validation"
            )

        if float(self.C) <= 0 or self._trial_C <= 0:
            raise ValueError("C and subproblem_C must be positive")
        if self.priority_strategy not in VALID_AUTHOR_VARIANTS:
            raise ValueError(
                f"priority_strategy must be one of {VALID_AUTHOR_VARIANTS}"
            )
        if self.insertion_strategy not in {"sequential", "binary_tree"}:
            raise ValueError(
                "insertion_strategy must be 'sequential' or 'binary_tree'"
            )

        self.history_: list[dict[str, Any]] = []
        self.priority_history_: list[dict[str, Any]] = []
        self.n_model_fits_ = 0
        self.n_batch_attempts_ = 0
        self.initial_core_repaired_ = False

        # This is the author's Initialsolution function made index-preserving.
        initial_model = self._make_svc(float(self.C))
        initial_model.fit(X_checked, encoded_y)
        initial_prediction = np.asarray(initial_model.predict(X_checked))
        initial_decision = np.asarray(
            initial_model.decision_function(X_checked)
        ).reshape(-1)
        initial_signed_margin = encoded_y * initial_decision
        core_mask = (initial_prediction == encoded_y) & (
            initial_signed_margin
            >= float(self.initial_margin) - float(self.feasibility_tolerance)
        )
        core_mask = self._ensure_two_classes(
            core_mask, encoded_y, initial_signed_margin
        )
        accepted = np.flatnonzero(core_mask).astype(int).tolist()

        current_model, feasible, initial_minimum = self._fit_trial(
            X_checked, encoded_y, accepted
        )
        if not feasible or current_model is None:
            if not self.repair_initial_core:
                raise RuntimeError(
                    "The initial core is not separable for the chosen kernel."
                )
            accepted, current_model, initial_minimum = self._repair_core(
                X_checked, encoded_y, accepted, initial_signed_margin
            )
            self.initial_core_repaired_ = True

        pool = sorted(set(range(len(X_checked))) - set(accepted))
        initial_pool = list(pool)
        feasibility_rejected: set[int] = set()
        self.initial_core_indices_ = np.asarray(sorted(accepted), dtype=int)
        self.initial_candidate_indices_ = np.asarray(initial_pool, dtype=int)
        self.initial_core_min_margin_ = float(initial_minimum)
        self.initial_ci_scores_: dict[int, float] = {}
        self.initial_ci_components_: dict[str, dict[int, float]] = {}

        master_iteration = 0
        while pool:
            master_iteration += 1
            pool_array = np.asarray(pool, dtype=int)
            decisions = np.asarray(
                current_model.decision_function(X_checked[pool_array])
            ).reshape(-1)
            predictions = np.asarray(current_model.predict(X_checked[pool_array]))
            signed_margins = encoded_y[pool_array] * decisions
            weak_mask = ~(
                (predictions == encoded_y[pool_array])
                & (signed_margins >= float(self.candidate_margin))
            )
            weak = pool_array[weak_mask].astype(int).tolist()
            weak_decisions = decisions[weak_mask]
            if not weak:
                break

            ci_result = self._candidate_priority(
                X_checked, encoded_y, weak, weak_decisions
            )
            order = np.argsort(-ci_result.scores, kind="stable")
            ordered = np.asarray(weak, dtype=int)[order].astype(int).tolist()
            score_map = {
                int(index): float(score)
                for index, score in zip(weak, ci_result.scores, strict=True)
            }
            component_map = {
                name: {
                    int(index): float(value)
                    for index, value in zip(weak, values, strict=True)
                }
                for name, values in ci_result.components.items()
            }
            self.priority_history_.append(
                {
                    "master_iteration": master_iteration,
                    "ordered_candidates": ordered,
                    "scores": score_map,
                    "components": component_map,
                }
            )
            if master_iteration == 1:
                self.initial_ci_scores_ = score_map
                self.initial_ci_components_ = component_map

            changed = False
            if self.insertion_strategy == "sequential":
                # Matches the upstream loop: failed points may be retried after
                # a different point changes the boundary.
                for candidate in ordered:
                    self.n_batch_attempts_ += 1
                    trial, is_feasible, minimum = self._fit_trial(
                        X_checked, encoded_y, accepted + [candidate]
                    )
                    event = {
                        "master_iteration": master_iteration,
                        "attempt": self.n_batch_attempts_,
                        "depth": 0,
                        "candidate_indices": [candidate],
                        "batch_size": 1,
                        "min_signed_margin": minimum,
                    }
                    if is_feasible and trial is not None:
                        accepted.append(candidate)
                        pool.remove(candidate)
                        feasibility_rejected.discard(candidate)
                        current_model = trial
                        event["action"] = "accepted"
                        self.history_.append(event)
                        changed = True
                        break
                    feasibility_rejected.add(candidate)
                    event["action"] = "failed_try"
                    self.history_.append(event)
            else:
                def try_block(block: list[int], depth: int) -> None:
                    nonlocal current_model, changed
                    if not block:
                        return
                    self.n_batch_attempts_ += 1
                    trial, is_feasible, minimum = self._fit_trial(
                        X_checked, encoded_y, accepted + block
                    )
                    event = {
                        "master_iteration": master_iteration,
                        "attempt": self.n_batch_attempts_,
                        "depth": depth,
                        "candidate_indices": list(block),
                        "batch_size": len(block),
                        "min_signed_margin": minimum,
                    }
                    if is_feasible and trial is not None:
                        accepted.extend(block)
                        for index in block:
                            if index in pool:
                                pool.remove(index)
                            feasibility_rejected.discard(index)
                        current_model = trial
                        event["action"] = "accepted"
                        self.history_.append(event)
                        changed = True
                        return
                    if len(block) == 1:
                        index = block[0]
                        if index in pool:
                            pool.remove(index)
                        feasibility_rejected.add(index)
                        event["action"] = "rejected"
                        self.history_.append(event)
                        changed = True
                        return
                    event["action"] = "split"
                    self.history_.append(event)
                    midpoint = math.ceil(len(block) / 2)
                    try_block(block[:midpoint], depth + 1)
                    try_block(block[midpoint:], depth + 1)

                if len(ordered) == 1:
                    try_block(ordered, 0)
                else:
                    root_midpoint = math.ceil(len(ordered) / 2)
                    try_block(ordered[:root_midpoint], 1)
                    try_block(ordered[root_midpoint:], 1)

            if not changed:
                break

        selected = sorted(set(accepted))
        not_selected = sorted(set(range(len(X_checked))) - set(selected))
        self.model_ = current_model
        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.rejected_indices_ = np.asarray(not_selected, dtype=int)
        self.feasibility_rejected_indices_ = np.asarray(
            sorted(feasibility_rejected - set(selected)), dtype=int
        )
        self.dormant_candidate_indices_ = np.asarray(
            sorted(set(not_selected) - set(self.feasibility_rejected_indices_)),
            dtype=int,
        )
        self.n_support_vectors_ = int(len(self.model_.support_))
        self.fit_time_seconds_ = float(time.perf_counter() - started)
        self.fit_summary_ = {
            "n_samples": int(len(X_checked)),
            "initial_core": int(len(self.initial_core_indices_)),
            "initial_candidates": int(len(self.initial_candidate_indices_)),
            "selected_total": int(len(self.selected_indices_)),
            "accepted_candidates": int(
                len(self.selected_indices_) - len(self.initial_core_indices_)
            ),
            "rejected_candidates": int(len(self.rejected_indices_)),
            "feasibility_rejected": int(
                len(self.feasibility_rejected_indices_)
            ),
            "dormant_candidates": int(len(self.dormant_candidate_indices_)),
            "support_vectors": self.n_support_vectors_,
            "master_iterations": int(master_iteration),
            "batch_attempts": int(self.n_batch_attempts_),
            "model_fits": int(self.n_model_fits_),
            "fit_time_seconds": self.fit_time_seconds_,
            "priority_strategy": self.priority_strategy,
            "insertion_strategy": self.insertion_strategy,
            "initial_core_repaired": bool(self.initial_core_repaired_),
        }
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        return np.asarray(
            self.model_.decision_function(check_array(X, dtype=float))
        )

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        encoded = np.asarray(
            self.model_.predict(check_array(X, dtype=float))
        )
        zero_one = np.where(encoded == -1, 0, 1)
        return self.label_encoder_.inverse_transform(zero_one)
