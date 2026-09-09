"""Candidate-priority functions for the BSVM master problem.

All functions return a *priority*: a larger value means that a candidate is
tested earlier.  This convention matches Algorithm 4 in the paper, whose
implementation uses ``w_y / |f(x)|`` and sorts in descending order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from sklearn.neighbors import NearestNeighbors


VALID_CI_STRATEGIES = (
    "paper_priority",
    "boundary_exp",
    "robust_hybrid",
    "user_formula_1",
    "user_formula_2",
)


@dataclass(frozen=True)
class CIResult:
    """Priority values and their normalized explanatory components."""

    scores: np.ndarray
    components: dict[str, np.ndarray]


def _class_terms(
    y: np.ndarray,
    candidate_indices: np.ndarray,
    class_weights: Mapping[int, float],
    eps: float,
) -> np.ndarray:
    raw = np.asarray([class_weights[int(y[i])] for i in candidate_indices], dtype=float)
    return raw / max(float(np.min(raw)), eps)


def _temperature(abs_decision: np.ndarray, supplied: Any, eps: float) -> float:
    if supplied is not None:
        value = float(supplied)
        if value <= 0:
            raise ValueError("ci_params['temperature'] must be positive")
        return value
    # A floor avoids a nearly singular exponential when all candidates sit on
    # the current boundary.
    return max(float(np.median(abs_decision)), 0.25, eps)


def _local_label_consistency(
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    if len(X) <= 1:
        return np.ones(len(candidate_indices), dtype=float)
    k = min(max(int(n_neighbors), 1), len(X) - 1)
    nearest = NearestNeighbors(n_neighbors=k + 1).fit(X)
    neighbor_indices = nearest.kneighbors(X[candidate_indices], return_distance=False)
    consistency = np.empty(len(candidate_indices), dtype=float)
    for row, (sample_index, neighbors) in enumerate(
        zip(candidate_indices, neighbor_indices, strict=True)
    ):
        usable = neighbors[neighbors != sample_index][:k]
        if len(usable) == 0:
            consistency[row] = 1.0
        else:
            consistency[row] = float(np.mean(y[usable] == y[sample_index]))
    return consistency


def _same_class_density_trust(
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    n_neighbors: int,
    eps: float,
) -> np.ndarray:
    """Return a bounded k-NN density proxy; isolated points approach zero."""

    trust = np.ones(len(candidate_indices), dtype=float)
    for label in np.unique(y):
        reference_indices = np.flatnonzero(y == label)
        positions = np.flatnonzero(y[candidate_indices] == label)
        if len(reference_indices) <= 1 or len(positions) == 0:
            continue

        k = min(max(int(n_neighbors), 1), len(reference_indices) - 1)
        nearest = NearestNeighbors(n_neighbors=k + 1).fit(X[reference_indices])

        # Establish a class-specific robust scale from typical within-class
        # neighbor distances. The first column is the sample itself.
        reference_distances = nearest.kneighbors(X[reference_indices], return_distance=True)[0]
        typical_distances = np.mean(reference_distances[:, 1 : k + 1], axis=1)
        scale = max(float(np.median(typical_distances)), eps)

        candidate_distances = nearest.kneighbors(
            X[candidate_indices[positions]], return_distance=True
        )[0]
        mean_distances = np.mean(candidate_distances[:, 1 : k + 1], axis=1)
        trust[positions] = 1.0 / (1.0 + mean_distances / scale)
    return trust


def _validation_gain_proxy(
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    validation_X: np.ndarray | None,
    validation_y: np.ndarray | None,
    n_neighbors: int,
) -> np.ndarray:
    """Compute the user-defined validation-gain proxy ``g_i``.

    ``g_i`` is the fraction of the candidate's k nearest validation samples
    whose label agrees with the candidate label.  Test data must never be
    passed here.
    """

    if validation_X is None or validation_y is None:
        raise ValueError(
            "user_formula_2 requires validation_X and validation_y; "
            "test data must not be used as a substitute"
        )
    validation_X = np.asarray(validation_X, dtype=float)
    validation_y = np.asarray(validation_y, dtype=int).reshape(-1)
    if validation_X.ndim != 2 or validation_X.shape[1] != X.shape[1]:
        raise ValueError("validation_X must be 2D with the same features as X")
    if len(validation_X) != len(validation_y) or len(validation_X) == 0:
        raise ValueError("validation_X and validation_y must be non-empty and aligned")
    k = min(max(int(n_neighbors), 1), len(validation_X))
    nearest = NearestNeighbors(n_neighbors=k).fit(validation_X)
    neighbor_indices = nearest.kneighbors(
        X[candidate_indices], return_distance=False
    )
    return np.mean(
        validation_y[neighbor_indices] == y[candidate_indices, None],
        axis=1,
    ).astype(float)


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Implement the monotone robust-ranking operator ``R[.]``.

    Only ordering matters to the master problem.  Rank log-products directly
    to preserve their ordering even when exponentiation would underflow.
    Stable sorting makes ties deterministic.
    """

    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) <= 1:
        return np.ones(len(values), dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks / float(len(values))


def _smoothing_floor(cfg: Mapping[str, Any], name: str) -> float:
    value = float(cfg.get(name, 0.0))
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"ci_params['{name}'] must be finite and in [0, 1]")
    return value


def _user_formula_priority(
    *,
    include_validation_gain: bool,
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    decision_values: np.ndarray,
    class_term: np.ndarray,
    validation_X: np.ndarray | None,
    validation_y: np.ndarray | None,
    cfg: Mapping[str, Any],
    eps: float,
) -> CIResult:
    """Compute either of the two user-supplied mathematical formulas."""

    local_floor = _smoothing_floor(cfg, "local_floor")
    gain_floor = (
        _smoothing_floor(cfg, "gain_floor") if include_validation_gain else 0.0
    )
    n_neighbors = int(cfg.get("n_neighbors", 7))
    local_raw = _local_label_consistency(X, y, candidate_indices, n_neighbors)
    local = np.clip(
        local_floor + (1.0 - local_floor) * local_raw,
        eps,
        1.0,
    )
    density = np.clip(
        _same_class_density_trust(X, y, candidate_indices, n_neighbors, eps),
        eps,
        1.0,
    )
    validation_gain = None
    validation_gain_effective = None
    if include_validation_gain:
        validation_gain = _validation_gain_proxy(
            X,
            y,
            candidate_indices,
            validation_X,
            validation_y,
            n_neighbors,
        )
        validation_gain_effective = np.clip(
            gain_floor + (1.0 - gain_floor) * validation_gain,
            eps,
            1.0,
        )

    signed_margin = y[candidate_indices] * decision_values
    margin_gap = np.abs(1.0 - signed_margin)
    supplied_tau = cfg.get("tau", cfg.get("temperature"))
    tau = _temperature(margin_gap, supplied_tau, eps)
    margin_target = np.exp(-margin_gap / tau)

    p = float(cfg.get("p", 1.0))
    beta = float(cfg.get("beta", 1.0))
    gamma = float(cfg.get("gamma", 1.0))
    delta = float(cfg.get("delta", 1.0))

    # Work in log space so large exponents cannot underflow before R[.].
    log_raw = (
        p * np.log(np.clip(class_term, eps, None))
        + beta * np.log(local)
        + gamma * np.log(density)
        - margin_gap / tau
    )
    if include_validation_gain:
        log_raw = log_raw + delta * np.log(validation_gain_effective)
    stabilized_raw = np.exp(log_raw - float(np.max(log_raw)))
    scores = _rank_normalize(log_raw)
    components = {
        "alpha_class": class_term,
        "r_local_consistency_raw": local_raw,
        "r_local_consistency": local,
        "rho_density": density,
        "signed_margin": signed_margin,
        "margin_target": margin_target,
        "raw_product_scaled": stabilized_raw,
        "R_rank": scores,
    }
    if include_validation_gain:
        components["g_validation_gain"] = validation_gain
        components["g_validation_gain_effective"] = validation_gain_effective
    return CIResult(scores=scores, components=components)


def compute_ci_priority(
    *,
    strategy: str,
    X: np.ndarray,
    y: np.ndarray,
    candidate_indices: np.ndarray,
    decision_values: np.ndarray,
    class_weights: Mapping[int, float],
    params: Mapping[str, Any] | None = None,
    validation_X: np.ndarray | None = None,
    validation_y: np.ndarray | None = None,
) -> CIResult:
    """Compute candidate priorities for one fitted decision boundary.

    Parameters
    ----------
    strategy:
        ``paper_priority`` reproduces Algorithm 4, ``boundary_exp`` replaces
        the reciprocal singularity with exponential decay,
        ``robust_hybrid`` discounts label-inconsistent and low-density points,
        and ``user_formula_1`` / ``user_formula_2`` implement the two supplied
        formulas centred on the unit signed margin.
    X, y:
        Complete numeric training set and encoded labels in ``{-1, +1}``.
    candidate_indices:
        Row indices for the current candidate set.
    decision_values:
        Decision-function values for ``candidate_indices`` in the same order.
    class_weights:
        Priority weights keyed by encoded label.
    params:
        Optional exponents and neighborhood sizes for the selected strategy.
        The user formulas optionally smooth local agreement with
        ``local_floor + (1 - local_floor) * r_i``.  ``user_formula_2`` also
        accepts ``gain_floor`` for the analogous validation-agreement factor.
        Both floors default to zero and must be finite values in [0, 1].
    validation_X, validation_y:
        Held-out validation data used only by ``user_formula_2`` to compute
        ``g_i``.  Test data must never be supplied.
    """

    if strategy not in VALID_CI_STRATEGIES:
        raise ValueError(
            f"Unknown ci_strategy={strategy!r}; choose one of {VALID_CI_STRATEGIES}"
        )

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    decision_values = np.asarray(decision_values, dtype=float).reshape(-1)
    if len(candidate_indices) != len(decision_values):
        raise ValueError("candidate_indices and decision_values must have equal length")

    cfg = dict(params or {})
    eps = float(cfg.get("eps", 1e-8))
    if eps <= 0:
        raise ValueError("ci_params['eps'] must be positive")

    abs_decision = np.abs(decision_values)
    class_term = _class_terms(y, candidate_indices, class_weights, eps)

    if strategy == "paper_priority":
        # Eq. (13) is |d_i| / |w_y|, while Algorithm 4 uses its reciprocal as
        # a descending priority. The epsilon fixes the division-by-zero spike.
        scores = class_term / (abs_decision + eps)
        return CIResult(
            scores=scores,
            components={"class": class_term, "inverse_boundary_distance": 1.0 / (abs_decision + eps)},
        )

    if strategy in {"user_formula_1", "user_formula_2"}:
        return _user_formula_priority(
            include_validation_gain=strategy == "user_formula_2",
            X=X,
            y=y,
            candidate_indices=candidate_indices,
            decision_values=decision_values,
            class_term=class_term,
            validation_X=validation_X,
            validation_y=validation_y,
            cfg=cfg,
            eps=eps,
        )

    temperature = _temperature(abs_decision, cfg.get("temperature"), eps)
    boundary = np.exp(-abs_decision / temperature)
    class_power = float(cfg.get("class_power", 1.0))
    boundary_power = float(cfg.get("boundary_power", 1.0))

    if strategy == "boundary_exp":
        scores = np.power(class_term, class_power) * np.power(boundary, boundary_power)
        return CIResult(
            scores=scores,
            components={"class": class_term, "boundary": boundary},
        )

    n_neighbors = int(cfg.get("n_neighbors", 7))
    local = _local_label_consistency(X, y, candidate_indices, n_neighbors)
    density = _same_class_density_trust(
        X, y, candidate_indices, n_neighbors, eps
    )
    local_floor = float(cfg.get("local_floor", 0.05))
    density_floor = float(cfg.get("density_floor", 0.05))
    local_power = float(cfg.get("local_power", 1.0))
    density_power = float(cfg.get("density_power", 0.5))

    local_term = np.clip(local_floor + (1.0 - local_floor) * local, eps, None)
    density_term = np.clip(
        density_floor + (1.0 - density_floor) * density, eps, None
    )
    scores = (
        np.power(class_term, class_power)
        * np.power(boundary, boundary_power)
        * np.power(local_term, local_power)
        * np.power(density_term, density_power)
    )
    return CIResult(
        scores=scores,
        components={
            "class": class_term,
            "boundary": boundary,
            "local_consistency": local,
            "density_trust": density,
        },
    )
