"""Reproducible, validation-selected extensions of the paper's result tables.

This runner deliberately does not change the legacy smoke-test runner.  Table
numbers describe an experiment plan, not a claim to reproduce published values.
Use ``--dry-run`` to inspect that plan before starting expensive kernel searches.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC, SVC
from threadpoolctl import threadpool_info


ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT / "src", ROOT / "examples"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import run_author_ci_comparison as legacy  # noqa: E402
from roch_bsvm import AuthorExtendedBSVMClassifier, VALID_AUTHOR_VARIANTS  # noqa: E402


NEW_VARIANTS = ("robust_hybrid", "user_formula_1", "user_formula_2")
BASELINES = ("soft_margin_svm", "weighted_svm", "nu_svm")
LABELS = {
    **legacy.VARIANT_LABELS,
    "soft_margin_svm": "Soft-margin SVM",
    "weighted_svm": "Weighted soft-margin SVM",
    "nu_svm": "Nu-SVM",
}
EXPERIMENT_1 = (
    ("dataset_4_labor", "labor", "experiment_1"),
    ("phpojxgl9", "ilpd", "experiment_1"),
    ("credit-approval", "credit-approval", "experiment_1"),
    ("fruitfly", "fruitfly", "experiment_1"),
    ("tecator", "tecator", "experiment_1"),
    ("quake", "quake", "experiment_1"),
    ("file19a81543501", "students-scores", "experiment_1"),
    ("titanic", "Titanic", "experiment_1"),
)
EXPERIMENT_2 = tuple(
    (stem, name, "experiment_2")
    for stem, (name, groups) in legacy.DATASET_PROTOCOLS.items()
    if "experiment_2" in groups
)
STEEL = (("php9xwopn", "steel-plates-fault", "industry_binary"),)
MULTICLASS = (
    ("dataset", "Fashion-MNIST", "multiclass"),
    ("dataset_41_glass", "glass", "multiclass"),
)


@dataclass(frozen=True)
class TableSpec:
    cases: tuple[tuple[str, str, str], ...]
    algorithms: tuple[str, ...]
    layout: str
    description: str


TABLE_SPECS = {
    "4": TableSpec(EXPERIMENT_1, BASELINES + ("author_original",) + NEW_VARIANTS,
                   "class_metrics", "Experiment 1: minority/majority classification"),
    "5": TableSpec(EXPERIMENT_2, ("soft_margin_svm", "nu_svm", "author_original") + NEW_VARIANTS,
                   "accuracy", "Experiment 2: classification performance"),
    "6": TableSpec(EXPERIMENT_1 + EXPERIMENT_2,
                   ("soft_margin_svm", "nu_svm", "author_original") + NEW_VARIANTS,
                   "efficiency", "Experiments 1 and 2: training, prediction and support vectors"),
    "c8": TableSpec(EXPERIMENT_1, ("author_original",) + NEW_VARIANTS,
                    "class_metrics", "Experiment 1: author BSVM and new priorities"),
    "c9": TableSpec(EXPERIMENT_1, ("soft_margin_svm",) + NEW_VARIANTS,
                    "class_metrics", "Experiment 1: soft-margin SVM and new priorities"),
    "c10": TableSpec(EXPERIMENT_2 + STEEL, ("author_original",) + NEW_VARIANTS,
                     "accuracy", "Experiment 2 and steel: author BSVM and new priorities"),
    "c11": TableSpec(EXPERIMENT_2 + STEEL, ("soft_margin_svm",) + NEW_VARIANTS,
                     "accuracy", "Experiment 2 and steel: soft-margin SVM and new priorities"),
    "c12": TableSpec(MULTICLASS, ("soft_margin_svm", "author_original") + NEW_VARIANTS,
                     "accuracy", "Multiclass: common one-vs-rest comparison"),
}
COMBO_KEYS = ("dataset", "paper_group", "kernel", "algorithm", "seed")
NOTES = [
    "Models are selected using validation only; the selected training fit is evaluated without refitting.",
    "Independent gain protocol holds reference-gain labels apart from model-selection validation labels.",
    "Nu-SVM uses balanced class weights in experiment_1 and equal weights elsewhere; infeasible nu trials are logged.",
    "Polynomial gamma is fixed to 1, the paper's (x dot z + coef0)^degree convention; legacy outputs used gamma=scale.",
    "Multiclass models all use one-vs-rest; support_vectors sums incidences across binary models, not unique samples.",
    "Selected fit time includes priority computation and OVR fits; tuning time and warmed prediction time are separate.",
    "total_model_fits includes the initial full-training SVC for each BSVM binary problem; legacy model_fits counts its trial fits only.",
    "Table means/std use only successful seeds; status and success counts expose missing or failed combinations.",
    "Default insertion policy changes both priority and candidate insertion; sequential/binary_tree overrides support ablation.",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_profile(path: Path, name: str) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if name not in document.get("profiles", {}):
        raise ValueError(f"Profile {name!r} is absent from {path}")
    profile = document["profiles"][name]
    if not isinstance(profile, dict):
        raise ValueError("A profile must map priority names to parameter objects")
    missing = set(VALID_AUTHOR_VARIANTS) - set(profile)
    if missing:
        raise ValueError(f"Profile {name!r} is missing priorities: {sorted(missing)}")
    for variant, params in profile.items():
        if variant not in VALID_AUTHOR_VARIANTS or not isinstance(params, dict):
            raise ValueError(f"Invalid profile entry: {variant!r}")
        validate_priority_params(variant, params)
    return profile


def validate_priority_params(variant: str, params: dict[str, Any]) -> None:
    common = {"eps", "n_neighbors"}
    allowed = {
        "author_original": {"eps"},
        "robust_hybrid": common | {"class_power", "boundary_power", "local_power", "density_power",
                                    "local_floor", "density_floor", "temperature"},
        "user_formula_1": common | {"p", "beta", "gamma", "tau", "temperature", "local_floor"},
        "user_formula_2": common | {"p", "beta", "gamma", "delta", "tau", "temperature", "local_floor", "gain_floor"},
    }
    unknown = set(params) - allowed[variant]
    if unknown:
        raise ValueError(f"{variant}: unknown or unused priority parameter(s) {sorted(unknown)}; "
                         f"allowed keys: {sorted(allowed[variant])}")
    if "tau" in params and "temperature" in params:
        raise ValueError(f"{variant}: use only one of tau or temperature; omit both for automatic scaling")
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"{variant}.{key} must be a finite JSON number")
        if key == "n_neighbors":
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{variant}.n_neighbors must be an integer >= 1")
        elif key in {"tau", "temperature", "eps"}:
            if value <= 0:
                hint = "; omit tau/temperature for automatic scaling (JSON does not use the legacy CLI's zero sentinel)" if key != "eps" else ""
                raise ValueError(f"{variant}.{key} must be positive{hint}")
        elif key.endswith("_floor"):
            if not 0 <= value <= 1:
                raise ValueError(f"{variant}.{key} must be in [0,1]")
        elif value < 0:
            raise ValueError(f"{variant}.{key} must be nonnegative")


def csv_inventory(data_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.csv")):
        key = path.stem.lower()
        if key == "manifest":
            continue
        if key in result:
            raise ValueError(f"Ambiguous CSV stems differing only in case: {result[key]}, {path}")
        result[key] = path
    return result


def select_cases(spec: TableSpec, requested: str | None) -> tuple[tuple[str, str, str], ...]:
    if not requested:
        return spec.cases
    names = {name.lower().removesuffix(".csv") for name in legacy.parse_list(requested) or ()}
    allowed = {name.lower() for stem, paper_name, _ in spec.cases for name in (stem, paper_name)}
    if names - allowed:
        raise ValueError(f"Datasets outside this table's scope: {sorted(names - allowed)}")
    return tuple(case for case in spec.cases if case[0].lower() in names or case[1].lower() in names)


def default_kernels(table: str) -> tuple[str, ...]:
    return ("rbf",) if table in {"4", "5", "6"} else legacy.DEFAULT_KERNELS


def parameter_grid(kernel: str, algorithm: str, search: str) -> list[dict[str, Any]]:
    """Use Table 3's C/degree/coef0/gamma grid and a separate nu grid."""
    grid = legacy.paper_grid(kernel, search)
    if algorithm == "nu_svm":
        nus = [0.1] if search == "none" else ([0.1, 0.75] if search == "fast" else [0.1, 0.75, 1.0])
        without_c = {canonical_json({k: v for k, v in params.items() if k != "C"})
                     for params in grid}
        grid = [dict(json.loads(params), nu=nu) for params in sorted(without_c) for nu in nus]
    if kernel == "poly":
        grid = [dict(params, gamma=1.0) for params in grid]
    return grid


def strict_split(indices: np.ndarray, y: np.ndarray, fraction: float, seed: int,
                 stage: str) -> tuple[np.ndarray, np.ndarray]:
    labels, counts = np.unique(y[indices], return_counts=True)
    n_right = int(np.ceil(len(indices) * fraction))
    if len(labels) < 2 or counts.min() < 2 or min(n_right, len(indices) - n_right) < len(labels):
        raise ValueError(
            f"{stage}: stratified split cannot preserve all {len(labels)} classes; "
            f"n={len(indices)}, counts={dict(zip(labels.tolist(), counts.tolist()))}, "
            f"requested sizes={len(indices) - n_right}/{n_right}. "
            "Use more data or explicitly choose --gain-protocol shared_validation if only the gain split is too small."
        )
    left, right = train_test_split(indices, test_size=fraction, random_state=seed,
                                   stratify=y[indices])
    if any(set(np.unique(y[part])) != set(labels) for part in (left, right)):
        raise ValueError(f"{stage}: at least one class disappeared after stratification")
    return np.asarray(left), np.asarray(right)


def split_indices(y: np.ndarray, *, seed: int, test_size: float = 0.2,
                  valid_size: float = 0.2, gain_protocol: str = "independent") -> dict[str, np.ndarray]:
    y = np.asarray(y)
    train_valid, test = strict_split(np.arange(len(y)), y, test_size, seed, "holdout test")
    train, validation = strict_split(train_valid, y, valid_size, seed, "training/validation")
    if gain_protocol == "independent":
        reference, select = strict_split(validation, y, 0.5, seed, "reference gain/selection validation")
    elif gain_protocol == "shared_validation":
        reference, select = validation.copy(), validation.copy()
    else:
        raise ValueError(f"Unknown gain protocol: {gain_protocol}")
    return {"train": train, "reference": reference, "validation": select, "test": test}


def prepare_data(path: Path, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if args.target_column not in frame:
        raise ValueError(f"Missing target column {args.target_column!r}")
    frame = frame.dropna(subset=[args.target_column]).reset_index(drop=True)
    frame[args.target_column] = frame[args.target_column].astype(str)
    original_n = len(frame)
    if 0 < args.max_rows < len(frame):
        y_all = frame[args.target_column].to_numpy()
        labels, counts = np.unique(y_all, return_counts=True)
        if counts.min() < 2 or min(args.max_rows, len(frame) - args.max_rows) < len(labels):
            raise ValueError("Subsampling cannot preserve all classes; increase --max-rows or use the full dataset")
        selected, _ = train_test_split(np.arange(len(frame)), train_size=args.max_rows,
                                       stratify=y_all, random_state=seed)
        frame = frame.iloc[selected].reset_index(drop=True)
    y = frame[args.target_column].to_numpy()
    indices = split_indices(y, seed=seed, test_size=args.test_size,
                            valid_size=args.valid_size, gain_protocol=args.gain_protocol)
    X = frame.drop(columns=[args.target_column])
    preprocessor = legacy.make_preprocessor(X.iloc[indices["train"]])
    data: dict[str, Any] = {"indices": indices, "n_samples": len(frame), "n_samples_source": original_n}
    data["X_train"] = np.asarray(preprocessor.fit_transform(X.iloc[indices["train"]]), dtype=float)
    for part in ("reference", "validation", "test"):
        data[f"X_{part}"] = np.asarray(preprocessor.transform(X.iloc[indices[part]]), dtype=float)
    for part, positions in indices.items():
        data[f"y_{part}"] = y[positions]
    data["n_features"] = data["X_train"].shape[1]
    data["split_sha256"] = hashlib.sha256(canonical_json({k: v.tolist() for k, v in indices.items()}).encode()).hexdigest()
    return data


def make_estimator(algorithm: str, kernel: str, params: dict[str, Any], group: str,
                   profile: dict[str, dict[str, Any]], args: argparse.Namespace, seed: int) -> Any:
    common = dict(kernel=kernel, gamma=params.get("gamma", "scale"),
                  degree=params.get("degree", 3), coef0=params.get("coef0", 0.0),
                  cache_size=args.cache_size, random_state=seed)
    if algorithm in BASELINES:
        weight = "balanced" if algorithm == "weighted_svm" or (algorithm == "nu_svm" and group == "experiment_1") else None
        if algorithm == "nu_svm":
            return NuSVC(nu=params["nu"], class_weight=weight, **common)
        return SVC(C=params["C"], class_weight=weight, **common)
    policy = args.insertion_policy
    if policy == "original":
        policy = "sequential" if algorithm == "author_original" else "binary_tree"
    return AuthorExtendedBSVMClassifier(
        C=params["C"], class_weight="balanced" if group == "experiment_1" else None,
        priority_strategy=algorithm, priority_params=dict(profile[algorithm]),
        insertion_strategy=policy, initial_margin=args.initial_margin,
        repair_initial_core=not args.no_core_repair, **common,
    )


def fit_model(estimator: Any, algorithm: str, data: dict[str, Any]) -> Any:
    """Only Formula 2 can read the reference labels; no fit sees selection/test."""
    X, y = data["X_train"], data["y_train"]
    labels = np.unique(y)

    def fit_one(binary_y: np.ndarray, binary_reference: np.ndarray | None = None) -> Any:
        fitted = clone(estimator)
        if algorithm == "user_formula_2":
            fitted.fit(X, binary_y, X_validation=data["X_reference"],
                       y_validation=data["y_reference"] if binary_reference is None else binary_reference)
        else:
            fitted.fit(X, binary_y)
        return fitted

    if len(labels) == 2:
        return fit_one(y)
    estimators = [fit_one(np.where(y == label, 1, 0),
                          np.where(data["y_reference"] == label, 1, 0)) for label in labels]
    return legacy.ManualOVRModel(labels, estimators)


def model_summary(model: Any, n_train: int) -> dict[str, Any]:
    estimators = getattr(model, "estimators_", [model])
    support = sum(len(est.model_.support_) if hasattr(est, "model_") else len(est.support_)
                  for est in estimators)
    is_bsvm = all(hasattr(est, "fit_summary_") for est in estimators)
    result = {"support_vectors": int(support), "binary_problems": len(estimators),
              "support_fraction": support / n_train,
              "support_fraction_per_binary_problem": support / (n_train * len(estimators))}
    if is_bsvm:
        result.update(legacy.aggregate_fit_summary(model))
        result["total_model_fits"] = result["model_fits"] + len(estimators)
    else:
        result.update(selected_total=n_train * len(estimators), accepted_candidates=0,
                      rejected_candidates=0, batch_attempts=0, model_fits=len(estimators),
                      total_model_fits=len(estimators), initial_core_repairs=0)
    return result


def class_labels(y_train: np.ndarray) -> tuple[str | None, str | None]:
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) != 2:
        return None, None
    order = np.argsort(counts, kind="stable")
    return str(labels[order[0]]), str(labels[order[-1]])


def metric_block(y: np.ndarray, prediction: np.ndarray, y_train: np.ndarray,
                 suffix: str) -> dict[str, Any]:
    minority, majority = class_labels(y_train)
    result: dict[str, Any] = legacy.metric_block(y, prediction, minority, suffix)
    result[f"majority_f1_{suffix}"] = (
        float(f1_score(y, prediction, pos_label=majority, zero_division=0)) if majority is not None else float("nan")
    )
    labels = np.unique(y_train)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, prediction, labels=labels, zero_division=0)
    per_class = {str(label): {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
                 for label, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)}
    result[f"per_class_{suffix}"] = canonical_json(per_class)
    if minority is not None:
        for label_type, label in (("minority", minority), ("majority", majority)):
            for metric in ("precision", "recall", "f1", "support"):
                result[f"{label_type}_{metric}_{suffix}"] = per_class[label][metric]
    return result


def selection_score(y: np.ndarray, prediction: np.ndarray, y_train: np.ndarray, group: str) -> float:
    minority, _ = class_labels(y_train)
    if group == "experiment_1":
        if minority is None:
            raise ValueError("Experiment 1 requires binary labels for minority-F1")
        return float(f1_score(y, prediction, pos_label=minority, zero_division=0))
    return float(accuracy_score(y, prediction))


def measured_prediction(model: Any, X: np.ndarray, repeats: int) -> tuple[np.ndarray, float, float]:
    prediction = model.predict(X)  # warm up the selected model and allocate prediction buffers
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        model.predict(X)
        durations.append(time.perf_counter() - started)
    median = float(np.median(durations))
    return prediction, median, median / len(X)


def tune_combination(combo: dict[str, Any], data: dict[str, Any], profile: dict[str, dict[str, Any]],
                     args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total_started = time.perf_counter()
    tuning: list[dict[str, Any]] = []
    best_key = (float("-inf"), float("-inf"), float("-inf"))
    best: tuple[Any, dict[str, Any], dict[str, Any], float, float, float] | None = None
    algorithm, group = combo["algorithm"], combo["paper_group"]
    for index, params in enumerate(parameter_grid(combo["kernel"], algorithm, args.search), start=1):
        started = time.perf_counter()
        trial = {**combo, "grid_index": index, "kernel_params": canonical_json(params),
                 "priority_params": canonical_json(profile.get(algorithm, {})),
                 "validation_metric": "minority_f1" if group == "experiment_1" else "accuracy"}
        try:
            estimator = make_estimator(algorithm, combo["kernel"], params, group, profile, args, combo["seed"])
            fit_started = time.perf_counter()
            model = fit_model(estimator, algorithm, data)
            fit_seconds = time.perf_counter() - fit_started
            prediction = model.predict(data["X_validation"])
            score = selection_score(data["y_validation"], prediction, data["y_train"], group)
            summary = model_summary(model, len(data["y_train"]))
            utility = score - (args.sv_penalty * summary["support_fraction"]
                               if args.selection_objective == "performance_sv" else 0.0)
            trial.update(status="ok", error="", validation_score=score, selection_utility=utility,
                         support_vectors=summary["support_vectors"], support_fraction=summary["support_fraction"],
                         selected_fit_seconds=fit_seconds, trial_seconds=time.perf_counter() - started)
            key = (utility, score, -summary["support_vectors"])
            if key > best_key:
                best_key = key
                best = (model, params, summary, fit_seconds, score, utility)
        except Exception as exc:
            trial.update(status="error", error=f"{type(exc).__name__}: {exc}",
                         trial_seconds=time.perf_counter() - started)
        tuning.append(trial)
    tuning_seconds = time.perf_counter() - total_started
    common = {**combo, "algorithm_label": LABELS[algorithm], "n_samples": data["n_samples"],
              "n_samples_source": data["n_samples_source"], "n_features": data["n_features"],
              "n_classes": len(np.unique(data["y_train"])),
              **{f"n_{part}": len(data[f"y_{part}"]) for part in ("train", "reference", "validation", "test")},
              "split_sha256": data["split_sha256"], "gain_protocol": args.gain_protocol,
              "profile": args.profile, "priority_params": canonical_json(profile.get(algorithm, {})),
              "selection_objective": args.selection_objective, "sv_penalty": args.sv_penalty,
              "validation_metric": "minority_f1" if group == "experiment_1" else "accuracy",
              "tuning_seconds": tuning_seconds, "grid_trials": len(tuning),
              "grid_errors": sum(row["status"] == "error" for row in tuning)}
    if best is None:
        return {**common, "status": "error", "error": tuning[-1]["error"] if tuning else "Empty grid"}, tuning
    model, params, summary, fit_seconds, score, utility = best
    minority, majority = class_labels(data["y_train"])
    row = {**common, **summary, "status": "ok", "error": "", "selected_params": canonical_json(params),
           "best_C": params.get("C"), "best_nu": params.get("nu"),
           "best_gamma": params.get("gamma", "scale"), "best_degree": params.get("degree", 3),
           "best_coef0": params.get("coef0", 0.0), "minority_label": minority or "",
           "majority_label": majority or "", "validation_score": score, "selection_utility": utility,
           "fit_time_seconds": fit_seconds, "prediction_repeats": args.prediction_repeats,
           "insertion_policy": (getattr(getattr(model, "estimators_", [model])[0], "insertion_strategy", "not_applicable"))}
    for part in ("train", "validation"):
        row.update(metric_block(data[f"y_{part}"], model.predict(data[f"X_{part}"]), data["y_train"], part))
    test_prediction, prediction_seconds, per_sample = measured_prediction(model, data["X_test"], args.prediction_repeats)
    row.update(metric_block(data["y_test"], test_prediction, data["y_train"], "test"))
    row["prediction_seconds"] = prediction_seconds
    row["prediction_seconds_per_sample"] = per_sample
    row["objective_test"] = selection_score(data["y_test"], test_prediction, data["y_train"], group)
    return row, tuning


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "status" not in results:
        return pd.DataFrame(columns=[*COMBO_KEYS, "baseline"])
    successful = results[results["status"] == "ok"]
    matching = ["dataset", "paper_group", "kernel", "seed"]
    metrics = ("objective_test", "accuracy_test", "balanced_accuracy_test", "f1_macro_test",
               "minority_f1_test", "majority_f1_test", "support_vectors", "total_model_fits",
               "fit_time_seconds", "prediction_seconds_per_sample", "tuning_seconds")
    rows = []
    for baseline in (*BASELINES, "author_original"):
        bases = successful[successful["algorithm"] == baseline].set_index(matching)
        for _, row in successful[successful["algorithm"].isin(NEW_VARIANTS)].iterrows():
            key = tuple(row[name] for name in matching)
            if key not in bases.index:
                continue
            original = bases.loc[key]
            entry = {name: row[name] for name in COMBO_KEYS}
            entry["baseline"] = baseline
            for metric in metrics:
                entry[f"{metric}_delta"] = row[metric] - original[metric]
            entry["support_reduction_percent"] = 100 * (1 - row["support_vectors"] / original["support_vectors"])
            entry["fit_speedup"] = original["fit_time_seconds"] / row["fit_time_seconds"]
            rows.append(entry)
    return pd.DataFrame(rows, columns=None if rows else [*COMBO_KEYS, "baseline"])


def wide_table(results: pd.DataFrame, table: str, kernels: tuple[str, ...]) -> pd.DataFrame:
    """Aggregate seeds; preserve status rows rather than hiding failed datasets."""
    if results.empty:
        return pd.DataFrame(columns=["dataset", "paper_group", "algorithm", "set", "class"])
    layout = TABLE_SPECS[table].layout
    keys = ["dataset", "paper_name", "paper_group", "algorithm"]
    rows = []
    for identity, group in results.groupby(keys, sort=False, dropna=False):
        if layout == "class_metrics":
            slices = [(part, kind, {metric: f"{kind}_{metric}_{part}" for metric in ("precision", "recall", "f1")})
                      for part in ("train", "test") for kind in ("minority", "majority")]
        elif layout == "efficiency":
            slices = [("selected_model", "all", {metric: metric for metric in (
                "fit_time_seconds", "prediction_seconds", "prediction_seconds_per_sample", "tuning_seconds",
                "support_vectors", "total_model_fits", "objective_test")})]
        else:
            slices = [(part, "all", {"accuracy": f"accuracy_{part}", "balanced_accuracy": f"balanced_accuracy_{part}",
                                    "macro_precision": f"precision_macro_{part}", "macro_recall": f"recall_macro_{part}",
                                    "macro_f1": f"f1_macro_{part}", "support_vectors": "support_vectors"})
                      for part in ("train", "test")]
        for part, kind, metrics in slices:
            entry = {**dict(zip(keys, identity, strict=True)), "set": part, "class": kind}
            for kernel in kernels:
                subset = group[group["kernel"] == kernel]
                ok = subset[subset["status"] == "ok"]
                entry[f"{kernel}__runs_ok"] = len(ok)
                entry[f"{kernel}__runs_total"] = len(subset)
                entry[f"{kernel}__status"] = ";".join(sorted(set(subset["status"].astype(str)))) or "not_run"
                for label, column in metrics.items():
                    values = pd.to_numeric(ok[column], errors="coerce").dropna() if column in ok else pd.Series(dtype=float)
                    entry[f"{kernel}__{label}_mean"] = values.mean() if len(values) else float("nan")
                    entry[f"{kernel}__{label}_std"] = values.std(ddof=1) if len(values) > 1 else float("nan")
            rows.append(entry)
    return pd.DataFrame(rows)


def combo_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(key) for key in COMBO_KEYS)


def validate_resume(output_dir: Path, config: dict[str, Any], resume: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config_file = output_dir / "run_config.json"
    existing = (output_dir / "results_long.csv").exists() or config_file.exists()
    if not resume:
        if existing:
            raise ValueError(f"Output directory already contains a run: {output_dir}. Use --resume with identical settings or a new directory.")
        return [], []
    if not config_file.exists():
        if existing:
            raise ValueError("Cannot resume results without run_config.json; use a new output directory")
        return [], []
    previous = json.loads(config_file.read_text(encoding="utf-8"))
    if previous.get("fingerprint") != config["fingerprint"]:
        raise ValueError("Resume configuration mismatch: settings, profile, source code or input data changed. Use a new output directory; existing results were not modified.")
    rows = legacy.read_records(output_dir / "results_long.csv")
    tuning = legacy.read_records(output_dir / "tuning_results.csv")
    if len({combo_key(row) for row in rows}) != len(rows):
        raise ValueError("Resume results contain duplicate combination keys; repair or use a new output directory")
    return rows, tuning


def run_configuration(args: argparse.Namespace, cases: tuple[tuple[str, str, str], ...],
                      kernels: tuple[str, ...], algorithms: tuple[str, ...], seeds: tuple[int, ...],
                      profile: dict[str, dict[str, Any]], inventory: dict[str, Path]) -> dict[str, Any]:
    settings = {key: value for key, value in vars(args).items() if key not in {"resume", "output_dir", "dry_run"}}
    inputs = {stem: {"path": str(inventory[stem]), "sha256": file_sha256(inventory[stem])}
              if stem in inventory else {"path": None, "sha256": None, "status": "missing_dataset"}
              for stem in sorted({case[0] for case in cases})}
    source_paths = (Path(__file__), ROOT / "examples/run_author_ci_comparison.py",
                    ROOT / "src/roch_bsvm/author_bsvm.py", ROOT / "src/roch_bsvm/scoring.py")
    signature = {"settings": settings, "effective_profile": profile,
                 "profile_sha256": file_sha256(resolve_path(args.profile_json)),
                 "cases": cases, "kernels": kernels, "algorithms": algorithms, "seeds": seeds,
                 "inputs": inputs, "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths},
                 "environment": {"python": sys.version, "python_executable": sys.executable,
                                 "numpy": np.__version__, "pandas": pd.__version__,
                                 "scikit_learn": sklearn.__version__, "platform": platform.platform(),
                                 "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                                 "threadpools": threadpool_info(),
                                 "thread_environment": {name: os.environ.get(name) for name in (
                                     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}}}
    return {**signature, "fingerprint": hashlib.sha256(canonical_json(signature).encode()).hexdigest(),
            "schema_version": 1, "notes": NOTES}


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], tuning: list[dict[str, Any]],
                  config: dict[str, Any], args: argparse.Namespace, kernels: tuple[str, ...], expected: int) -> None:
    comparison = pd.DataFrame(rows)
    wide = wide_table(comparison, args.table, kernels)
    comparison.to_csv(output_dir / "results_long.csv", index=False)
    pd.DataFrame(tuning, columns=None if tuning else [*COMBO_KEYS, "grid_index", "status", "error"]).to_csv(output_dir / "tuning_results.csv", index=False)
    paired_deltas(comparison).to_csv(output_dir / "paired_deltas.csv", index=False)
    wide.to_csv(output_dir / "table.csv", index=False)
    statuses = comparison["status"].value_counts().to_dict() if "status" in comparison else {}
    errors = [{key: row.get(key) for key in (*COMBO_KEYS, "status", "error")}
              for row in rows if row.get("status") != "ok"]
    metadata = {"table": args.table, "expected_combinations": expected,
                "recorded_combinations": len(rows), "status_counts": statuses,
                "requested_plan_complete": len(rows) == expected and statuses.get("ok", 0) == expected,
                "full_paper_scope_requested": args.datasets is None and args.kernels is None and args.algorithms is None,
                "missing_datasets": sorted({row["dataset"] for row in rows if row.get("status") == "missing_dataset"}),
                "errors": errors, "fingerprint": config["fingerprint"], "notes": NOTES}
    (output_dir / "completeness.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    message = [f"# Extended paper table {args.table.upper()}", "", TABLE_SPECS[args.table].description,
               "", f"Successful combinations: {statuses.get('ok', 0)}/{expected}. Profile: {args.profile}; gain protocol: {args.gain_protocol}.",
               "", "Scores use a 0–1 scale. Means/std aggregate seeds; a single seed has no estimated standard deviation.",
               "", legacy.markdown_table(wide), "", "Protocol notes:", "", *[f"- {note}" for note in NOTES]]
    if errors:
        message.extend(["", "Missing or failed combinations:", "", legacy.markdown_table(pd.DataFrame(errors))])
    (output_dir / "table.md").write_text("\n".join(message) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True, type=str.lower, choices=tuple(TABLE_SPECS))
    parser.add_argument("--data-dir", default="data/csv")
    parser.add_argument("--datasets", help="Subset of this table's CSV stems or paper dataset names")
    parser.add_argument("--kernels", help="Subset of linear,poly,sigmoid,rbf; default: rbf for 4/5/6, all four for appendix tables")
    parser.add_argument("--algorithms", help="Subset of the table's algorithms; c9/c11 additionally allow author_original")
    parser.add_argument("--search", choices=("none", "fast", "paper"), default="paper")
    parser.add_argument("--profile-json", default="config/ci_profiles.json")
    parser.add_argument("--profile", default="conservative")
    parser.add_argument("--insertion-policy", choices=("original", "sequential", "binary_tree"), default="original")
    parser.add_argument("--gain-protocol", choices=("independent", "shared_validation"), default="independent")
    parser.add_argument("--selection-objective", choices=("paper", "performance_sv"), default="paper")
    parser.add_argument("--sv-penalty", type=float, default=0.05)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--initial-margin", type=float, default=1.0)
    parser.add_argument("--no-core-repair", action="store_true")
    parser.add_argument("--cache-size", type=float, default=200.0)
    parser.add_argument("--prediction-repeats", type=int, default=20)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if not 0 < args.test_size < 1 or not 0 < args.valid_size < 1:
        raise ValueError("--test-size and --valid-size must be in (0,1)")
    if args.prediction_repeats < 1 or args.max_rows < 0 or args.sv_penalty < 0 or args.cache_size <= 0:
        raise ValueError("Require prediction-repeats>=1, max-rows>=0, sv-penalty>=0 and cache-size>0")
    seeds = tuple(int(value) for value in legacy.parse_list(args.seeds) or ())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--seeds must contain distinct integers")
    spec = TABLE_SPECS[args.table]
    cases = select_cases(spec, args.datasets)
    if not cases:
        raise ValueError("The requested dataset subset is empty")
    kernels = legacy.parse_list(args.kernels) or default_kernels(args.table)
    if set(kernels) - set(legacy.DEFAULT_KERNELS) or len(kernels) != len(set(kernels)):
        raise ValueError("--kernels must be a distinct subset of linear,poly,sigmoid,rbf")
    algorithms = legacy.parse_list(args.algorithms) or spec.algorithms
    allowed = set(spec.algorithms) | ({"author_original"} if args.table in {"c9", "c11"} else set())
    if set(algorithms) - allowed or len(algorithms) != len(set(algorithms)):
        raise ValueError(f"--algorithms must be a distinct subset of {sorted(allowed)}")
    profile = load_profile(resolve_path(args.profile_json), args.profile)
    inventory = csv_inventory(resolve_path(args.data_dir))
    config = run_configuration(args, cases, kernels, algorithms, seeds, profile, inventory)
    expected = len(cases) * len(kernels) * len(algorithms) * len(seeds)
    plan = {"table": args.table, "description": spec.description, "cases": cases, "kernels": kernels,
            "algorithms": algorithms, "seeds": seeds, "profile": args.profile, "gain_protocol": args.gain_protocol,
            "expected_combinations": expected,
            "grid_trials_per_dataset_seed": sum(len(parameter_grid(kernel, algorithm, args.search))
                                                for kernel, algorithm in itertools.product(kernels, algorithms)),
            "missing_datasets": sorted({stem for stem, _, _ in cases if stem not in inventory}),
            "fingerprint": config["fingerprint"]}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0
    output_dir = resolve_path(args.output_dir or f"outputs/paper_table_{args.table}")
    rows, tuning = validate_resume(output_dir, config, args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Commit provenance before any fit, so an interrupted first combination is resumable.
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    for stem, paper_name, group in cases:
        for seed in seeds:
            combinations = [{"table": args.table, "dataset": stem, "paper_name": paper_name,
                             "paper_group": group, "kernel": kernel, "algorithm": algorithm, "seed": seed}
                            for kernel, algorithm in itertools.product(kernels, algorithms)]
            # Missing data and deterministic split errors are completed plan
            # records for this fingerprint.  A changed input file changes the
            # fingerprint, so a newly supplied dataset is still rerun safely.
            complete_statuses = {"ok", "missing_dataset", "split_error", "data_error"}
            complete = {combo_key(row) for row in rows if row.get("status") in complete_statuses}
            pending = [combo for combo in combinations if combo_key(combo) not in complete]
            if not pending:
                print(f"[resume] completed dataset={stem} group={group} seed={seed}", flush=True)
                continue
            data, failure = None, None
            if stem not in inventory:
                failure = ("missing_dataset", f"CSV for {stem!r} is unavailable in {resolve_path(args.data_dir)}")
            else:
                try:
                    data = prepare_data(inventory[stem], args, seed)
                except Exception as exc:
                    failure = ("split_error" if "split" in str(exc).lower() or "stratif" in str(exc).lower()
                               else "data_error", f"{type(exc).__name__}: {exc}")
            if failure:
                print(f"[warning] dataset={stem} group={group} seed={seed}: {failure[1]}", flush=True)
            for combo in pending:
                key = combo_key(combo)
                rows = [row for row in rows if combo_key(row) != key]
                tuning = [row for row in tuning if combo_key(row) != key]
                if failure:
                    row, trials = ({**combo, "algorithm_label": LABELS[combo["algorithm"]],
                                    "status": failure[0], "error": failure[1]}, [])
                else:
                    print(f"[fit] dataset={stem} group={group} seed={seed} kernel={combo['kernel']} algorithm={combo['algorithm']}", flush=True)
                    row, trials = tune_combination(combo, data, profile, args)
                    if row["status"] != "ok":
                        print(f"[warning] {row['error']}", flush=True)
                rows.append(row)
                row["run_fingerprint"] = config["fingerprint"]
                for trial in trials:
                    trial["run_fingerprint"] = config["fingerprint"]
                tuning.extend(trials)
                write_outputs(output_dir, rows, tuning, config, args, kernels, expected)
    write_outputs(output_dir, rows, tuning, config, args, kernels, expected)
    statuses = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    successful = statuses.get("ok", 0)
    print(f"Completed combinations: {successful}/{expected}; statuses={statuses}", flush=True)
    if successful != expected:
        print(f"[warning] INCOMPLETE TABLE: {expected - successful} requested combinations are missing or failed. "
              f"See {output_dir / 'completeness.json'}; blank scores are not experimental results.", flush=True)
    print(f"Wrote {output_dir / 'table.md'}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
