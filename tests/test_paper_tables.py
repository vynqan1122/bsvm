"""Protocol and export regression checks for the separate paper-table runner."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import run_paper_table as runner  # noqa: E402


@pytest.fixture
def profile():
    return {
        "author_original": {"eps": 1e-8},
        "robust_hybrid": {"n_neighbors": 3, "density_power": 0.5},
        "user_formula_1": {"n_neighbors": 3, "gamma": 0.5},
        "user_formula_2": {"n_neighbors": 3, "gamma": 0.5, "delta": 0.5},
    }


@pytest.fixture
def prepared(tmp_path):
    X, y = make_classification(n_samples=160, n_features=4, n_informative=3,
                               n_redundant=0, weights=[0.65, 0.35],
                               class_sep=1.5, flip_y=0, random_state=17)
    frame = pd.DataFrame(X, columns=list("abcd"))
    frame["target"] = y.astype(str)
    path = tmp_path / "fruitfly.csv"
    frame.to_csv(path, index=False)
    args = runner.build_parser().parse_args(["--table", "4", "--search", "none",
                                            "--prediction-repeats", "2"])
    return runner.prepare_data(path, args, 42), args, path


def combination(algorithm="soft_margin_svm"):
    return {"table": "4", "dataset": "fruitfly", "paper_name": "fruitfly",
            "paper_group": "experiment_1", "kernel": "linear", "algorithm": algorithm, "seed": 42}


def test_independent_split_preserves_all_classes_without_overlap():
    y = np.array(["majority"] * 120 + ["minority"] * 80)
    parts = runner.split_indices(y, seed=42)
    assert {name: len(rows) for name, rows in parts.items()} == {
        "train": 128, "reference": 16, "validation": 16, "test": 40}
    assert len(set(np.concatenate(list(parts.values())))) == len(y)
    for indices in parts.values():
        assert set(y[indices]) == {"majority", "minority"}
    shared = runner.split_indices(y, seed=42, gain_protocol="shared_validation")
    assert len(shared["train"]) == 128
    assert np.array_equal(shared["reference"], shared["validation"])
    assert len(shared["reference"]) == 32
    with pytest.raises(ValueError, match="stratified split"):
        runner.split_indices(np.array(["a"] * 8 + ["b"] * 7), seed=42)


def test_formula_two_receives_reference_only(prepared):
    data, _, _ = prepared
    captured = []

    class RecordingEstimator(BaseEstimator):
        def fit(self, X, y, **kwargs):
            captured.append((X.copy(), y.copy(), kwargs))
            return self

    runner.fit_model(RecordingEstimator(), "user_formula_2", data)
    X_seen, y_seen, kwargs = captured[-1]
    np.testing.assert_array_equal(X_seen, data["X_train"])
    np.testing.assert_array_equal(y_seen, data["y_train"])
    np.testing.assert_array_equal(kwargs["X_validation"], data["X_reference"])
    np.testing.assert_array_equal(kwargs["y_validation"], data["y_reference"])
    assert not np.array_equal(kwargs["X_validation"], data["X_validation"])
    for algorithm in ("author_original", "user_formula_1", "soft_margin_svm"):
        runner.fit_model(RecordingEstimator(), algorithm, data)
        assert captured[-1][2] == {}


@pytest.mark.parametrize("algorithm", runner.BASELINES)
def test_real_sklearn_baselines_export_metrics_and_timing(prepared, profile, algorithm):
    data, args, _ = prepared
    row, trials = runner.tune_combination(combination(algorithm), data, profile, args)
    assert row["status"] == "ok", row.get("error")
    assert len(trials) == 1 and trials[0]["status"] == "ok"
    assert row["n_train"] + row["n_reference"] + row["n_validation"] + row["n_test"] == row["n_samples"]
    assert 0 < row["support_vectors"] <= row["n_train"]
    assert row["total_model_fits"] == 1
    assert row["fit_time_seconds"] > 0
    assert row["prediction_seconds_per_sample"] > 0
    assert row["tuning_seconds"] >= row["fit_time_seconds"]
    assert row["objective_test"] == pytest.approx(row["minority_f1_test"])
    assert row["majority_f1_test"] == pytest.approx(2 * row["f1_macro_test"] - row["minority_f1_test"])
    classes = json.loads(row["per_class_test"])
    assert sum(item["support"] for item in classes.values()) == row["n_test"]
    assert row["majority_label"] != row["minority_label"]
    wide = runner.wide_table(pd.DataFrame([row]), "4", ("linear",))
    assert set(wide["class"]) == {"minority", "majority"}
    exported = wide[(wide["set"] == "test") & (wide["class"] == "majority")].iloc[0]
    assert exported["linear__f1_mean"] == pytest.approx(row["majority_f1_test"])
    assert pd.isna(exported["linear__f1_std"])


def test_validation_selects_model_even_when_test_prefers_other_candidate(prepared, profile, monkeypatch):
    data, args, _ = prepared
    minority, majority = runner.class_labels(data["y_train"])
    data["y_validation"] = np.full(len(data["y_validation"]), minority)
    data["y_test"] = np.full(len(data["y_test"]), majority)

    class ConstantClassifier(BaseEstimator):
        def __init__(self, label):
            self.label = label

        def fit(self, X, y):
            self.support_ = np.array([0, 1])
            return self

        def predict(self, X):
            return np.full(len(X), self.label)

    monkeypatch.setattr(runner, "parameter_grid", lambda *unused: [{"C": 1.0}, {"C": 10.0}])
    monkeypatch.setattr(runner, "make_estimator", lambda algorithm, kernel, params, *unused:
                        ConstantClassifier(minority if params["C"] == 1.0 else majority))
    row, trials = runner.tune_combination(combination(), data, profile, args)
    assert row["best_C"] == 1.0
    assert row["validation_score"] == 1.0
    assert row["accuracy_test"] == 0.0
    assert [trial["validation_score"] for trial in trials] == [1.0, 0.0]


def test_metrics_use_training_class_identity_even_if_test_balance_flips():
    y_train = np.array(["A", "A", "A", "B"])
    y_test = np.array(["A", "B", "B", "B"])
    prediction = np.array(["A", "A", "B", "B"])
    metrics = runner.metric_block(y_test, prediction, y_train, "test")
    assert metrics["minority_f1_test"] == pytest.approx(f1_score(y_test, prediction, pos_label="B"))
    assert metrics["majority_f1_test"] == pytest.approx(f1_score(y_test, prediction, pos_label="A"))
    assert metrics["accuracy_test"] == pytest.approx(accuracy_score(y_test, prediction))
    assert runner.class_labels(np.array(["A", "B"])) == ("A", "B")


def test_paper_grids_use_poly_convention_and_do_not_mix_c_with_nu():
    for table in ("4", "5", "6"):
        assert runner.default_kernels(table) == ("rbf",)
    for table in ("c8", "c9", "c10", "c11", "c12"):
        assert set(runner.default_kernels(table)) == {"linear", "poly", "sigmoid", "rbf"}
    assert len(runner.parameter_grid("linear", "soft_margin_svm", "paper")) == 4
    assert len(runner.parameter_grid("rbf", "author_original", "paper")) == 16
    poly = runner.parameter_grid("poly", "user_formula_1", "paper")
    assert len(poly) == 48 and all(params["gamma"] == 1.0 for params in poly)
    nu = runner.parameter_grid("rbf", "nu_svm", "paper")
    assert len(nu) == 12
    assert {params["nu"] for params in nu} == {0.1, 0.75, 1.0}
    assert all("C" not in params for params in nu)


def test_resume_refuses_changed_settings_without_overwriting_results(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    config = {"fingerprint": "first", "settings": {"profile": "legacy"}}
    (output / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame([{**combination(), "status": "ok", "best_C": 10.0}]).to_csv(output / "results_long.csv", index=False)
    rows, tuning = runner.validate_resume(output, config, True)
    assert len(rows) == 1 and not tuning
    before = (output / "results_long.csv").read_bytes()
    with pytest.raises(ValueError, match="Resume configuration mismatch"):
        runner.validate_resume(output, {"fingerprint": "different"}, True)
    assert (output / "results_long.csv").read_bytes() == before
    assert json.loads((output / "run_config.json").read_text()) == config
    with pytest.raises(ValueError, match="already contains a run"):
        runner.validate_resume(output, config, False)


def test_fingerprint_detects_profile_and_input_data_changes(prepared, profile, tmp_path):
    _, args, csv_path = prepared
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"profiles": {"conservative": profile}}), encoding="utf-8")
    args.profile_json = str(profile_path)
    cases = (("fruitfly", "fruitfly", "experiment_1"),)
    inventory = {"fruitfly": csv_path}

    def fingerprint():
        return runner.run_configuration(args, cases, ("linear",), ("author_original",),
                                        (42,), profile, inventory)["fingerprint"]

    first = fingerprint()
    args.resume, args.output_dir = True, "somewhere_else"
    assert fingerprint() == first
    with csv_path.open("a", encoding="utf-8") as source:
        source.write("\n")
    changed_data = fingerprint()
    assert changed_data != first
    profile["user_formula_1"]["gamma"] = 1.0
    assert fingerprint() != changed_data


@pytest.mark.parametrize("variant,params,match", [
    ("robust_hybrid", {"beta": 1.0}, "unknown or unused"),
    ("user_formula_1", {"delta": 1.0}, "unknown or unused"),
    ("user_formula_1", {"tau": 0}, "omit tau/temperature"),
    ("user_formula_2", {"tau": 1, "temperature": 2}, "only one"),
    ("robust_hybrid", {"n_neighbors": 2.5}, "integer"),
    ("user_formula_2", {"gain_floor": 1.1}, "in \\[0,1\\]"),
    ("user_formula_1", {"beta": -1}, "nonnegative"),
    ("user_formula_1", {"gamma": float("nan")}, "finite"),
    ("author_original", {"eps": 0}, "positive"),
])
def test_profile_rejects_ignored_or_invalid_parameters(tmp_path, profile, variant, params, match):
    profile[variant] = params
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"profiles": {"conservative": profile}}), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        runner.load_profile(path, "conservative")


def test_nu_infeasible_trials_are_logged_without_discarding_success(prepared, profile):
    data, args, _ = prepared
    args.search = "paper"
    row, trials = runner.tune_combination(combination("nu_svm"), data, profile, args)
    assert row["status"] == "ok"
    assert len(trials) == 3
    failures = [trial for trial in trials if trial["status"] == "error"]
    assert failures and all(trial["error"] for trial in failures)
    assert row["grid_errors"] == len(failures)


def test_missing_dataset_is_visible_in_table_and_completeness(tmp_path, profile):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"profiles": {"conservative": profile}}), encoding="utf-8")
    output = tmp_path / "outputs"
    exit_code = runner.main(["--table", "4", "--data-dir", str(tmp_path),
                             "--datasets", "credit-approval", "--kernels", "linear",
                             "--algorithms", "author_original,user_formula_1",
                             "--profile-json", str(profile_path), "--output-dir", str(output)])
    assert exit_code == 0
    long = pd.read_csv(output / "results_long.csv")
    assert len(long) == 2 and set(long["status"]) == {"missing_dataset"}
    wide = pd.read_csv(output / "table.csv")
    assert set(wide["linear__status"]) == {"missing_dataset"}
    assert wide["linear__f1_mean"].isna().all()
    metadata = json.loads((output / "completeness.json").read_text())
    assert metadata["missing_datasets"] == ["credit-approval"]
    assert metadata["expected_combinations"] == 2
    assert not metadata["requested_plan_complete"]


def test_paired_deltas_require_matching_protocol_kernel_and_seed(prepared, profile):
    data, args, _ = prepared
    original, _ = runner.tune_combination(combination(), data, profile, args)
    variant = {**original, "algorithm": "user_formula_1", "objective_test": original["objective_test"] + 0.1,
               "support_vectors": original["support_vectors"] - 1}
    unmatched = {**variant, "seed": 43}
    deltas = runner.paired_deltas(pd.DataFrame([original, variant, unmatched]))
    assert len(deltas) == 1
    assert deltas.iloc[0]["objective_test_delta"] == pytest.approx(0.1)
    assert deltas.iloc[0]["support_vectors_delta"] == -1


def test_multiclass_baseline_is_ovr_and_exports_macro_metrics(prepared, profile):
    data, args, _ = prepared
    # Three well separated deterministic classes, with every partition represented.
    for part, n in (("train", 45), ("reference", 9), ("validation", 9), ("test", 12)):
        y = np.resize(np.array(["a", "b", "c"]), n)
        data[f"y_{part}"] = y
        data[f"X_{part}"] = np.column_stack([np.searchsorted(["a", "b", "c"], y) * 4,
                                             np.linspace(-0.1, 0.1, n)])
    data["n_samples"], data["n_samples_source"], data["n_features"] = 75, 75, 2
    combo = {**combination(), "table": "c12", "paper_group": "multiclass"}
    row, _ = runner.tune_combination(combo, data, profile, args)
    assert row["status"] == "ok"
    assert row["binary_problems"] == 3 and row["total_model_fits"] == 3
    assert np.isfinite(row["f1_macro_test"])
    assert np.isnan(row["minority_f1_test"]) and np.isnan(row["majority_f1_test"])
    wide = runner.wide_table(pd.DataFrame([row]), "c12", ("linear",))
    test_row = wide[wide["set"] == "test"].iloc[0]
    assert test_row["linear__macro_precision_mean"] == pytest.approx(row["precision_macro_test"])
    assert test_row["linear__macro_recall_mean"] == pytest.approx(row["recall_macro_test"])
    assert test_row["linear__support_vectors_mean"] == row["support_vectors"]
    efficiency = runner.wide_table(pd.DataFrame([row]), "6", ("linear",)).iloc[0]
    assert efficiency["linear__prediction_seconds_mean"] == row["prediction_seconds"]
    assert efficiency["linear__tuning_seconds_mean"] == row["tuning_seconds"]
