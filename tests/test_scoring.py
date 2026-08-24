import numpy as np
import pytest

from roch_bsvm.scoring import compute_ci_priority


def test_robust_hybrid_downweights_far_isolated_candidate():
    X = np.array(
        [
            [-2.0, 0.0],
            [-1.8, 0.1],
            [-1.6, -0.1],
            [1.6, 0.0],
            [1.8, 0.1],
            [2.0, -0.1],
            [0.05, 0.0],
            [12.0, 12.0],
        ]
    )
    y = np.array([-1, -1, -1, 1, 1, 1, 1, 1])
    candidates = np.array([6, 7])
    result = compute_ci_priority(
        strategy="robust_hybrid",
        X=X,
        y=y,
        candidate_indices=candidates,
        decision_values=np.array([0.05, -8.0]),
        class_weights={-1: 1.0, 1: 1.0},
        params={"n_neighbors": 2},
    )
    assert np.isfinite(result.scores).all()
    assert result.scores[0] > result.scores[1]


def test_paper_priority_is_finite_on_boundary():
    result = compute_ci_priority(
        strategy="paper_priority",
        X=np.array([[-1.0], [1.0]]),
        y=np.array([-1, 1]),
        candidate_indices=np.array([0]),
        decision_values=np.array([0.0]),
        class_weights={-1: 2.0, 1: 1.0},
        params={"eps": 1e-6},
    )
    assert np.isfinite(result.scores[0])


def test_user_formula_1_targets_the_unit_signed_margin():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    candidates = np.array([1, 2])
    # Signed margins are 0.9 and 0.2, so candidate 1 is closer to m_i = 1.
    result = compute_ci_priority(
        strategy="user_formula_1",
        X=X,
        y=y,
        candidate_indices=candidates,
        decision_values=np.array([-0.9, 0.2]),
        class_weights={-1: 1.0, 1: 1.0},
        params={"n_neighbors": 1, "tau": 1.0},
    )
    assert result.components["signed_margin"].tolist() == pytest.approx([0.9, 0.2])
    assert result.components["margin_target"][0] > result.components["margin_target"][1]
    assert result.scores[0] > result.scores[1]


def test_user_formula_2_uses_validation_label_agreement():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([-1, -1, 1, 1])
    candidates = np.array([1, 2])
    validation_X = np.array([[-1.1], [-0.9], [0.9], [1.1]])
    # The left candidate agrees with nearby validation labels; the right one does not.
    validation_y = np.array([-1, -1, -1, -1])
    result = compute_ci_priority(
        strategy="user_formula_2",
        X=X,
        y=y,
        candidate_indices=candidates,
        decision_values=np.array([-0.5, 0.5]),
        class_weights={-1: 1.0, 1: 1.0},
        params={"n_neighbors": 2, "tau": 1.0},
        validation_X=validation_X,
        validation_y=validation_y,
    )
    assert result.components["g_validation_gain"].tolist() == pytest.approx([1.0, 0.0])
    assert result.scores[0] > result.scores[1]


def test_user_formula_2_rejects_missing_validation_data():
    with pytest.raises(ValueError, match="validation_X"):
        compute_ci_priority(
            strategy="user_formula_2",
            X=np.array([[-1.0], [1.0]]),
            y=np.array([-1, 1]),
            candidate_indices=np.array([0]),
            decision_values=np.array([-0.5]),
            class_weights={-1: 1.0, 1: 1.0},
        )
