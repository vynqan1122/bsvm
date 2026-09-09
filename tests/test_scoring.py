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


@pytest.mark.parametrize("strategy", ["user_formula_1", "user_formula_2"])
def test_user_formula_rank_preserves_order_when_products_underflow(strategy):
    result = compute_ci_priority(
        strategy=strategy,
        X=np.array([[0.0], [1.0], [2.0]]),
        y=np.ones(3, dtype=int),
        candidate_indices=np.arange(3),
        decision_values=np.array([801.0, 1001.0, 1.0]),
        class_weights={1: 1.0},
        params={"p": 0, "beta": 0, "gamma": 0, "delta": 0, "tau": 1.0},
        validation_X=np.array([[0.0]]),
        validation_y=np.array([1]),
    )
    # exp(-800) and exp(-1000) both round to zero, but their mathematical
    # priorities are distinct. Candidate order must not break this ranking.
    np.testing.assert_array_equal(result.components["raw_product_scaled"], [0, 0, 1])
    np.testing.assert_allclose(result.scores, [2 / 3, 1 / 3, 1])


@pytest.mark.parametrize("strategy", ["user_formula_1", "user_formula_2"])
def test_local_floor_can_retain_a_zero_agreement_candidate(strategy):
    inputs = dict(
        strategy=strategy,
        X=np.array([[0.0], [-0.1], [0.1], [10.0], [9.9]]),
        y=np.array([1, -1, -1, 1, 1]),
        candidate_indices=np.array([0, 3]),
        decision_values=np.array([1.0, -4.0]),
        class_weights={-1: 1.0, 1: 1.0},
        validation_X=np.array([[0.0], [10.0]]),
        validation_y=np.array([1, 1]),
    )
    params = {"n_neighbors": 1, "gamma": 0, "tau": 1.0}
    default = compute_ci_priority(**inputs, params=params)
    zero = compute_ci_priority(**inputs, params={**params, "local_floor": 0.0})
    smoothed = compute_ci_priority(**inputs, params={**params, "local_floor": 0.05})
    disabled = compute_ci_priority(**inputs, params={**params, "local_floor": 1.0})

    np.testing.assert_array_equal(default.scores, zero.scores)
    np.testing.assert_array_equal(default.components["raw_product_scaled"], zero.components["raw_product_scaled"])
    assert default.scores[0] < default.scores[1]
    assert smoothed.scores[0] > smoothed.scores[1]
    np.testing.assert_array_equal(smoothed.components["r_local_consistency_raw"], [0, 1])
    np.testing.assert_allclose(smoothed.components["r_local_consistency"], [0.05, 1])
    assert np.isfinite(smoothed.components["raw_product_scaled"]).all()
    np.testing.assert_allclose(disabled.components["r_local_consistency"], [1, 1])


def test_gain_floor_can_retain_a_zero_validation_agreement_candidate():
    inputs = dict(
        strategy="user_formula_2",
        X=np.array([[0.0], [10.0]]),
        y=np.array([1, 1]),
        candidate_indices=np.array([0, 1]),
        decision_values=np.array([1.0, -4.0]),
        class_weights={1: 1.0},
        validation_X=np.array([[0.0], [10.0]]),
        validation_y=np.array([-1, 1]),
    )
    params = {"n_neighbors": 1, "beta": 0, "gamma": 0, "tau": 1.0}
    default = compute_ci_priority(**inputs, params=params)
    zero = compute_ci_priority(**inputs, params={**params, "gain_floor": 0.0})
    smoothed = compute_ci_priority(**inputs, params={**params, "gain_floor": 0.05})
    disabled = compute_ci_priority(**inputs, params={**params, "gain_floor": 1.0})

    np.testing.assert_array_equal(default.scores, zero.scores)
    np.testing.assert_array_equal(default.components["raw_product_scaled"], zero.components["raw_product_scaled"])
    assert default.scores[0] < default.scores[1]
    assert smoothed.scores[0] > smoothed.scores[1]
    np.testing.assert_array_equal(smoothed.components["g_validation_gain"], [0, 1])
    np.testing.assert_allclose(smoothed.components["g_validation_gain_effective"], [0.05, 1])
    assert np.isfinite(smoothed.components["raw_product_scaled"]).all()
    np.testing.assert_allclose(disabled.components["g_validation_gain_effective"], [1, 1])


@pytest.mark.parametrize(
    "strategy,parameter",
    [
        ("user_formula_1", "local_floor"),
        ("user_formula_2", "local_floor"),
        ("user_formula_2", "gain_floor"),
    ],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf, -np.inf])
def test_user_formula_rejects_invalid_smoothing_floor(strategy, parameter, value):
    with pytest.raises(ValueError, match=parameter):
        compute_ci_priority(
            strategy=strategy,
            X=np.array([[-1.0], [1.0]]),
            y=np.array([-1, 1]),
            candidate_indices=np.array([0]),
            decision_values=np.array([-0.5]),
            class_weights={-1: 1.0, 1: 1.0},
            params={parameter: value},
            validation_X=np.array([[-1.0], [1.0]]),
            validation_y=np.array([-1, 1]),
        )
