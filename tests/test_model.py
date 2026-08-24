import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from roch_bsvm import (
    AuthorExtendedBSVMClassifier,
    BinaryTreeBSVMClassifier,
    tune_ci_strategy,
)


def test_binary_tree_classifier_partitions_candidates_and_predicts():
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        weights=[0.8, 0.2],
        class_sep=1.2,
        flip_y=0.08,
        random_state=7,
    )
    model = BinaryTreeBSVMClassifier(
        kernel="linear",
        C=1.0,
        subproblem_C=1000.0,
        initial_margin=1.1,
        ci_strategy="robust_hybrid",
        ci_params={"n_neighbors": 5},
    ).fit(X, y)

    prediction = model.predict(X[:11])
    assert prediction.shape == (11,)
    assert set(prediction).issubset(set(y))
    handled = set(model.selected_indices_) | set(model.rejected_indices_)
    assert handled == set(range(len(X)))
    assert not (set(model.selected_indices_) & set(model.rejected_indices_))
    assert model.fit_summary_["support_vectors"] == model.n_support_vectors_


def test_binary_batching_accepts_separable_points_in_few_attempts():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [rng.normal(-2.0, 0.2, size=(20, 2)), rng.normal(2.0, 0.2, size=(20, 2))]
    )
    y = np.array([0] * 20 + [1] * 20)
    model = BinaryTreeBSVMClassifier(
        kernel="linear",
        C=1.0,
        subproblem_C=1000.0,
        initial_margin=3.0,
        ci_strategy="boundary_exp",
    ).fit(X, y)
    assert len(model.initial_candidate_indices_) >= 4
    assert len(model.rejected_indices_) == 0
    assert model.n_batch_attempts_ < len(model.initial_candidate_indices_)


def test_sequential_insertion_is_available_as_original_like_baseline():
    X, y = make_classification(
        n_samples=70,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        weights=[0.75, 0.25],
        random_state=13,
    )
    model = BinaryTreeBSVMClassifier(
        kernel="linear",
        C=1.0,
        subproblem_C=1000.0,
        initial_margin=1.2,
        ci_strategy="paper_priority",
        ci_params={"eps": 1e-6},
        insertion_strategy="sequential",
    ).fit(X, y)

    assert model.fit_summary_["insertion_strategy"] == "sequential"
    assert model.n_batch_attempts_ == len(model.initial_candidate_indices_)


def test_ci_tuning_returns_ranked_results():
    X, y = make_classification(
        n_samples=80,
        n_features=4,
        weights=[0.7, 0.3],
        random_state=11,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=11
    )
    model, results = tune_ci_strategy(
        X_train,
        y_train,
        X_valid,
        y_valid,
        base_estimator=BinaryTreeBSVMClassifier(kernel="linear", subproblem_C=1000),
        candidates=(
            {"ci_strategy": "paper_priority", "ci_params": {"eps": 1e-6}},
            {"ci_strategy": "boundary_exp", "ci_params": {}},
        ),
    )
    assert len(results) == 2
    assert results[0]["validation_score"] >= results[1]["validation_score"]
    assert hasattr(model, "model_")


def test_author_extended_formula_2_uses_held_out_validation():
    X, y = make_classification(
        n_samples=90,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        weights=[0.7, 0.3],
        random_state=21,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=21
    )
    model = AuthorExtendedBSVMClassifier(
        kernel="linear",
        C=10.0,
        priority_strategy="user_formula_2",
        priority_params={"n_neighbors": 3, "tau": 1.0},
        insertion_strategy="binary_tree",
    ).fit(
        X_train,
        y_train,
        X_validation=X_valid,
        y_validation=y_valid,
    )
    assert model.predict(X_valid[:7]).shape == (7,)
    assert model.fit_summary_["priority_strategy"] == "user_formula_2"
    assert len(model.selected_indices_) + len(model.rejected_indices_) == len(X_train)
