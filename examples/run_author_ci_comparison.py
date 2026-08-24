"""Compare the author's BSVM with three new c_i variants on local datasets.

The split, validation objective and paper hyperparameter grid follow Sections
3.7-3.8 of Mohasel & Koosha (Neurocomputing 671, 2026).  Results are flushed
after every dataset/group/kernel/variant so ``--resume`` can continue a run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roch_bsvm import (  # noqa: E402
    AuthorExtendedBSVMClassifier,
    VALID_AUTHOR_VARIANTS,
)


DEFAULT_KERNELS = ("linear", "poly", "sigmoid", "rbf")
VARIANT_LABELS = {
    "author_original": "Author original BSVM",
    "robust_hybrid": "New 1 - robust_hybrid",
    "user_formula_1": "New 2 - user_formula_1",
    "user_formula_2": "New 3 - user_formula_2",
}

# Local CSV stem -> (paper dataset name, paper experiment groups).
DATASET_PROTOCOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "dataset_4_labor": ("labor", ("experiment_1",)),
    "phpojxgl9": ("ilpd", ("experiment_1",)),
    "fruitfly": ("fruitfly", ("experiment_1", "experiment_2")),
    "tecator": ("tecator", ("experiment_1", "experiment_2")),
    "quake": ("quake", ("experiment_1",)),
    "file19a81543501": ("students-scores", ("experiment_1",)),
    "titanic": ("Titanic", ("experiment_1",)),
    "sleuth_case2002": ("sleuth-case2002", ("experiment_2",)),
    "leukemia": ("leukemia", ("experiment_2",)),
    "cloud": ("cloud", ("experiment_2",)),
    "php90oy2b": ("aids", ("experiment_2",)),
    "prnn_synth": ("prnn-synth", ("experiment_2",)),
    "dataset_114_shuttle-landing-control": (
        "shuttle-landing-control",
        ("experiment_2",),
    ),
    "rabe_266": ("rabe-266", ("experiment_2",)),
    "php9xwopn": ("steel-plates-fault", ("industry_binary",)),
    "dataset": ("Fashion-MNIST", ("multiclass",)),
    "dataset_41_glass": ("glass", ("multiclass",)),
}


def parse_list(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def discover_csv_files(
    data_dir: Path, datasets: tuple[str, ...] | None
) -> list[Path]:
    files = sorted(
        path for path in data_dir.glob("*.csv") if path.name.lower() != "manifest.csv"
    )
    if not datasets:
        return files
    requested = {item.lower() for item in datasets}
    selected = [
        path
        for path in files
        if path.stem.lower() in requested or path.name.lower() in requested
    ]
    found = {
        name
        for path in selected
        for name in (path.stem.lower(), path.name.lower())
    }
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"Datasets not found in {data_dir}: {missing}")
    return selected


def protocol_cases(csv_path: Path, selection: str) -> list[tuple[str, str]]:
    paper_name, groups = DATASET_PROTOCOLS.get(
        csv_path.stem.lower(), (csv_path.stem, ("experiment_2",))
    )
    if selection == "auto":
        return [(paper_name, groups[0])]
    if selection == "all":
        return [(paper_name, group) for group in groups]
    if selection in groups:
        return [(paper_name, selection)]
    return []


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = frame.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in frame.columns if column not in numeric_columns
    ]
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ],
        sparse_threshold=0.0,
    )


def maybe_subsample(
    frame: pd.DataFrame,
    target_column: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    y = frame[target_column].astype(str)
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and max_rows >= len(counts) else None
    sampled, _ = train_test_split(
        frame,
        train_size=max_rows,
        stratify=stratify,
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def safe_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    counts = y.value_counts()
    n_test = int(np.ceil(len(y) * test_size))
    stratify = y if counts.min() >= 2 and n_test >= len(counts) else None
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )


def paper_grid(kernel: str, search: str) -> list[dict[str, Any]]:
    if search == "none":
        values = {
            "C": [10.0],
            "degree": [3],
            "coef0": [0.0],
            "gamma": ["scale"],
        }
    elif search == "fast":
        values = {
            "C": [1.0, 10.0],
            "degree": [2, 3],
            "coef0": [0.0],
            "gamma": [0.01, 0.1],
        }
    else:
        values = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "degree": [2, 3, 4, 5],
            "coef0": [0.0, 0.5, 1.0],
            "gamma": [0.001, 0.01, 0.1, 1.0],
        }
    keys = ["C"]
    if kernel == "poly":
        keys += ["degree", "coef0"]
    elif kernel == "sigmoid":
        keys += ["gamma", "coef0"]
    elif kernel == "rbf":
        keys += ["gamma"]
    return [
        dict(zip(keys, combination, strict=True))
        for combination in itertools.product(*(values[key] for key in keys))
    ]


def protocol_settings(group: str) -> tuple[str, str | None]:
    if group == "experiment_1":
        return "minority_f1", "balanced"
    return "accuracy", None


class ManualOVRModel:
    """Small OVR wrapper that forwards validation labels to every binary fit."""

    def __init__(self, classes: np.ndarray, estimators: list[Any]) -> None:
        self.classes_ = np.asarray(classes)
        self.estimators_ = estimators

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.column_stack(
            [np.asarray(estimator.decision_function(X)).reshape(-1) for estimator in self.estimators_]
        )
        return self.classes_[np.argmax(scores, axis=1)]


def build_estimator(
    *,
    variant: str,
    kernel: str,
    kernel_params: dict[str, Any],
    class_weight: str | None,
    args: argparse.Namespace,
) -> AuthorExtendedBSVMClassifier:
    if variant == "author_original":
        insertion_strategy = "sequential"
        priority_params: dict[str, Any] = {"eps": args.eps}
    elif variant == "robust_hybrid":
        insertion_strategy = "binary_tree"
        priority_params = {
            "eps": args.eps,
            "n_neighbors": args.n_neighbors,
            "class_power": args.p,
            "boundary_power": args.beta,
            "local_power": args.gamma_power,
            "density_power": args.delta,
        }
        if args.tau > 0:
            priority_params["temperature"] = args.tau
    else:
        insertion_strategy = "binary_tree"
        priority_params = {
            "eps": args.eps,
            "n_neighbors": args.n_neighbors,
            "p": args.p,
            "beta": args.beta,
            "gamma": args.gamma_power,
            "delta": args.delta,
        }
        if args.tau > 0:
            priority_params["tau"] = args.tau
    return AuthorExtendedBSVMClassifier(
        kernel=kernel,
        gamma=kernel_params.get("gamma", "scale"),
        degree=int(kernel_params.get("degree", 3)),
        coef0=float(kernel_params.get("coef0", 0.0)),
        C=float(kernel_params["C"]),
        subproblem_C=None,
        class_weight=class_weight,
        priority_strategy=variant,
        priority_params=priority_params,
        insertion_strategy=insertion_strategy,
        initial_margin=args.initial_margin,
        candidate_margin=1.0,
        repair_initial_core=not args.no_core_repair,
        random_state=args.random_state,
    )


def fit_binary_or_ovr(
    estimator: AuthorExtendedBSVMClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> Any:
    classes = np.unique(y_train)
    if len(classes) == 2:
        return clone(estimator).fit(
            X_train,
            y_train,
            X_validation=X_valid,
            y_validation=y_valid,
        )
    estimators = []
    for label in classes:
        binary_train = np.where(y_train == label, 1, 0)
        binary_valid = np.where(y_valid == label, 1, 0)
        fitted = clone(estimator).fit(
            X_train,
            binary_train,
            X_validation=X_valid,
            y_validation=binary_valid,
        )
        estimators.append(fitted)
    return ManualOVRModel(classes, estimators)


def aggregate_fit_summary(model: Any) -> dict[str, Any]:
    estimators = (
        [model]
        if hasattr(model, "fit_summary_")
        else [
            estimator
            for estimator in getattr(model, "estimators_", [])
            if hasattr(estimator, "fit_summary_")
        ]
    )
    sum_keys = (
        "selected_total",
        "accepted_candidates",
        "rejected_candidates",
        "support_vectors",
        "batch_attempts",
        "model_fits",
        "fit_time_seconds",
    )
    summary = {key: sum(est.fit_summary_[key] for est in estimators) for key in sum_keys}
    summary["binary_problems"] = len(estimators)
    summary["initial_core_repairs"] = sum(
        int(est.fit_summary_["initial_core_repaired"]) for est in estimators
    )
    return summary


def objective_score(
    metric: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    minority_label: str | None,
) -> float:
    if metric == "minority_f1":
        return float(
            f1_score(
                y_true,
                prediction,
                pos_label=minority_label,
                zero_division=0,
            )
        )
    return float(accuracy_score(y_true, prediction))


def metric_block(
    y_true: np.ndarray,
    prediction: np.ndarray,
    minority_label: str | None,
    suffix: str,
) -> dict[str, float]:
    result = {
        f"accuracy_{suffix}": float(accuracy_score(y_true, prediction)),
        f"balanced_accuracy_{suffix}": float(
            balanced_accuracy_score(y_true, prediction)
        ),
        f"precision_macro_{suffix}": float(
            precision_score(y_true, prediction, average="macro", zero_division=0)
        ),
        f"recall_macro_{suffix}": float(
            recall_score(y_true, prediction, average="macro", zero_division=0)
        ),
        f"f1_macro_{suffix}": float(
            f1_score(y_true, prediction, average="macro", zero_division=0)
        ),
    }
    result[f"minority_f1_{suffix}"] = (
        float(
            f1_score(
                y_true,
                prediction,
                pos_label=minority_label,
                zero_division=0,
            )
        )
        if minority_label is not None
        else float("nan")
    )
    return result


def tune_and_evaluate(
    *,
    csv_stem: str,
    paper_name: str,
    paper_group: str,
    kernel: str,
    variant: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int,
    n_features: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validation_metric, class_weight = protocol_settings(paper_group)
    labels, counts = np.unique(y_train, return_counts=True)
    minority_label = (
        str(labels[int(np.argmin(counts))]) if len(labels) == 2 else None
    )
    tuning_rows: list[dict[str, Any]] = []
    best_model = None
    best_config = None
    best_validation = float("-inf")
    best_utility = float("-inf")
    best_support_vectors = np.inf
    total_started = time.perf_counter()

    for grid_index, kernel_params in enumerate(paper_grid(kernel, args.search), start=1):
        started = time.perf_counter()
        try:
            estimator = build_estimator(
                variant=variant,
                kernel=kernel,
                kernel_params=kernel_params,
                class_weight=class_weight,
                args=args,
            )
            model = fit_binary_or_ovr(
                estimator, X_train, y_train, X_valid, y_valid
            )
            validation_prediction = model.predict(X_valid)
            validation_score = objective_score(
                validation_metric,
                y_valid,
                validation_prediction,
                minority_label,
            )
            summary = aggregate_fit_summary(model)
            support_fraction = float(summary["support_vectors"] / len(X_train))
            utility = validation_score
            if args.selection_objective == "performance_sv":
                utility -= float(args.sv_penalty) * support_fraction
            tuning_rows.append(
                {
                    "dataset": csv_stem,
                    "paper_name": paper_name,
                    "paper_group": paper_group,
                    "kernel": kernel,
                    "variant": variant,
                    "grid_index": grid_index,
                    "status": "ok",
                    "error": "",
                    "kernel_params": json.dumps(kernel_params, sort_keys=True),
                    "validation_metric": validation_metric,
                    "validation_score": validation_score,
                    "support_vectors": summary["support_vectors"],
                    "support_fraction": support_fraction,
                    "selection_utility": utility,
                    "fit_seconds": time.perf_counter() - started,
                }
            )
            candidate_key = (utility, validation_score, -summary["support_vectors"])
            best_key = (best_utility, best_validation, -best_support_vectors)
            if candidate_key > best_key:
                best_model = model
                best_config = dict(kernel_params)
                best_validation = validation_score
                best_utility = utility
                best_support_vectors = summary["support_vectors"]
        except Exception as exc:
            tuning_rows.append(
                {
                    "dataset": csv_stem,
                    "paper_name": paper_name,
                    "paper_group": paper_group,
                    "kernel": kernel,
                    "variant": variant,
                    "grid_index": grid_index,
                    "status": "error",
                    "error": str(exc),
                    "kernel_params": json.dumps(kernel_params, sort_keys=True),
                    "validation_metric": validation_metric,
                    "fit_seconds": time.perf_counter() - started,
                }
            )

    if best_model is None or best_config is None:
        errors = [row["error"] for row in tuning_rows if row["status"] == "error"]
        raise RuntimeError(errors[-1] if errors else "No valid hyperparameter configuration")

    train_prediction = best_model.predict(X_train)
    test_prediction = best_model.predict(X_test)
    summary = aggregate_fit_summary(best_model)
    row: dict[str, Any] = {
        "dataset": csv_stem,
        "paper_name": paper_name,
        "paper_group": paper_group,
        "kernel": kernel,
        "variant": variant,
        "algorithm": VARIANT_LABELS[variant],
        "status": "ok",
        "error": "",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": len(labels),
        "binary_problems": summary["binary_problems"],
        "validation_metric": validation_metric,
        "validation_score": best_validation,
        "selection_objective": args.selection_objective,
        "selection_utility": best_utility,
        "minority_label": minority_label or "",
        "priority_strategy": variant,
        "insertion_strategy": (
            "sequential" if variant == "author_original" else "binary_tree"
        ),
        "best_C": best_config.get("C"),
        "best_gamma": best_config.get("gamma", "scale"),
        "best_degree": best_config.get("degree", 3),
        "best_coef0": best_config.get("coef0", 0.0),
        "selected_total": summary["selected_total"],
        "accepted_candidates": summary["accepted_candidates"],
        "rejected_candidates": summary["rejected_candidates"],
        "support_vectors": summary["support_vectors"],
        "batch_attempts": summary["batch_attempts"],
        "model_fits": summary["model_fits"],
        "initial_core_repairs": summary["initial_core_repairs"],
        "fit_time_seconds": summary["fit_time_seconds"],
        "elapsed_tuning_seconds": time.perf_counter() - total_started,
    }
    row.update(metric_block(y_train, train_prediction, minority_label, "train"))
    row.update(metric_block(y_test, test_prediction, minority_label, "test"))
    row["objective_test"] = objective_score(
        validation_metric, y_test, test_prediction, minority_label
    )
    return row, tuning_rows


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("-")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_delta_table(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "status" not in results.columns:
        return pd.DataFrame()
    ok = results[results["status"] == "ok"].copy()
    keys = ["dataset", "paper_group", "kernel"]
    baseline = ok[ok["variant"] == "author_original"].set_index(keys)
    new_rows = ok[ok["variant"] != "author_original"].set_index(keys)
    metrics = (
        "objective_test",
        "accuracy_test",
        "balanced_accuracy_test",
        "f1_macro_test",
        "minority_f1_test",
        "support_vectors",
        "model_fits",
        "elapsed_tuning_seconds",
    )
    rows = []
    for index, row in new_rows.iterrows():
        if index not in baseline.index:
            continue
        base = baseline.loc[index]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[-1]
        output = dict(zip(keys, index, strict=True))
        output["variant"] = row["variant"]
        output["algorithm"] = row["algorithm"]
        for metric in metrics:
            output[f"{metric}_delta"] = row[metric] - base[metric]
        rows.append(output)
    return pd.DataFrame(rows)


def build_summary(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty:
        return pd.DataFrame()
    rows = []
    for variant, group in deltas.groupby("variant", sort=False):
        objective = group["objective_test_delta"].dropna()
        support = group["support_vectors_delta"].dropna()
        rows.append(
            {
                "variant": variant,
                "algorithm": VARIANT_LABELS.get(variant, variant),
                "comparisons": len(group),
                "objective_mean_delta": objective.mean(),
                "objective_wins": int((objective > 0).sum()),
                "objective_ties": int(np.isclose(objective, 0).sum()),
                "objective_losses": int((objective < 0).sum()),
                "support_vectors_mean_delta": support.mean(),
                "fewer_support_vectors": int((support < 0).sum()),
                "model_fits_mean_delta": group["model_fits_delta"].mean(),
            }
        )
    return pd.DataFrame(rows)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except EmptyDataError:
        return []


def remove_combo_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    paper_group: str,
    kernel: str,
    variant: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if not (
            row.get("dataset") == dataset
            and row.get("paper_group") == paper_group
            and row.get("kernel") == kernel
            and row.get("variant") == variant
        )
    ]


def combo_complete(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    paper_group: str,
    kernel: str,
    variant: str,
) -> bool:
    return any(
        row.get("dataset") == dataset
        and row.get("paper_group") == paper_group
        and row.get("kernel") == kernel
        and row.get("variant") == variant
        and row.get("status") == "ok"
        for row in rows
    )


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    tuning_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    comparison = pd.DataFrame(rows)
    tuning = pd.DataFrame(tuning_rows)
    deltas = build_delta_table(comparison)
    summary = build_summary(deltas)
    comparison.to_csv(output_dir / "comparison_all.csv", index=False)
    tuning.to_csv(output_dir / "tuning_results.csv", index=False)
    deltas.to_csv(output_dir / "comparison_deltas_vs_original.csv", index=False)
    summary.to_csv(output_dir / "summary_by_variant.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    compact_columns = [
        column
        for column in (
            "dataset",
            "paper_group",
            "kernel",
            "algorithm",
            "status",
            "validation_score",
            "objective_test",
            "accuracy_test",
            "f1_macro_test",
            "minority_f1_test",
            "support_vectors",
            "model_fits",
            "error",
        )
        if column in comparison.columns
    ]
    (output_dir / "comparison_report.md").write_text(
        "\n".join(
            [
                "# Author BSVM vs three new c_i functions",
                "",
                "## Aggregate deltas versus author_original",
                "",
                markdown_table(summary),
                "",
                "## Per-combination deltas",
                "",
                markdown_table(deltas),
                "",
                "## Full results (compact view)",
                "",
                markdown_table(comparison[compact_columns] if compact_columns else comparison),
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/csv")
    parser.add_argument("--datasets", default=None, help="Comma-separated CSV stems")
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument(
        "--variants", default=",".join(VALID_AUTHOR_VARIANTS)
    )
    parser.add_argument(
        "--paper-groups",
        choices=["all", "auto", "experiment_1", "experiment_2"],
        default="all",
        help="all repeats fruitfly/tecator under both paper protocols",
    )
    parser.add_argument("--search", choices=["none", "fast", "paper"], default="fast")
    parser.add_argument("--selection-objective", choices=["paper", "performance_sv"], default="paper")
    parser.add_argument("--sv-penalty", type=float, default=0.05)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--output-dir", default="outputs/author_ci_comparison")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma-power", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.0, help="0 = robust automatic scale")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--initial-margin", type=float, default=1.0)
    parser.add_argument("--no-core-repair", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-flush-each", action="store_true")
    args = parser.parse_args()

    kernels = parse_list(args.kernels) or DEFAULT_KERNELS
    variants = parse_list(args.variants) or VALID_AUTHOR_VARIANTS
    unknown_variants = sorted(set(variants) - set(VALID_AUTHOR_VARIANTS))
    if unknown_variants:
        raise SystemExit(f"Unknown variants: {unknown_variants}")
    unknown_kernels = sorted(set(kernels) - set(DEFAULT_KERNELS))
    if unknown_kernels:
        raise SystemExit(f"Unknown kernels: {unknown_kernels}")

    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = discover_csv_files(data_dir, parse_list(args.datasets))
    rows = read_records(output_dir / "comparison_all.csv") if args.resume else []
    tuning_rows = read_records(output_dir / "tuning_results.csv") if args.resume else []

    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        if args.target_column not in frame.columns:
            print(f"[skip] {csv_path.name}: missing target column", flush=True)
            continue
        frame = frame.dropna(subset=[args.target_column]).reset_index(drop=True)
        frame[args.target_column] = frame[args.target_column].astype(str)
        frame = maybe_subsample(
            frame, args.target_column, args.max_rows, args.random_state
        )
        X = frame.drop(columns=[args.target_column])
        y = frame[args.target_column]
        if y.nunique() < 2:
            print(f"[skip] {csv_path.name}: fewer than two classes", flush=True)
            continue

        for paper_name, paper_group in protocol_cases(csv_path, args.paper_groups):
            X_train_valid, X_test, y_train_valid, y_test = safe_split(
                X,
                y,
                test_size=args.test_size,
                random_state=args.random_state,
            )
            X_train, X_valid, y_train, y_valid = safe_split(
                X_train_valid,
                y_train_valid,
                test_size=args.valid_size,
                random_state=args.random_state,
            )
            preprocessor = make_preprocessor(X_train)
            Xt_train = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
            Xt_valid = np.asarray(preprocessor.transform(X_valid), dtype=float)
            Xt_test = np.asarray(preprocessor.transform(X_test), dtype=float)
            yt_train = np.asarray(y_train.astype(str))
            yt_valid = np.asarray(y_valid.astype(str))
            yt_test = np.asarray(y_test.astype(str))

            for kernel in kernels:
                for variant in variants:
                    combo = dict(
                        dataset=csv_path.stem,
                        paper_group=paper_group,
                        kernel=kernel,
                        variant=variant,
                    )
                    if args.resume and combo_complete(rows, **combo):
                        print(
                            f"[skip] dataset={csv_path.stem} group={paper_group} "
                            f"kernel={kernel} variant={variant}",
                            flush=True,
                        )
                        continue
                    rows = remove_combo_rows(rows, **combo)
                    tuning_rows = remove_combo_rows(tuning_rows, **combo)
                    print(
                        f"[run] dataset={csv_path.stem} group={paper_group} "
                        f"kernel={kernel} variant={variant}",
                        flush=True,
                    )
                    try:
                        row, tuning = tune_and_evaluate(
                            csv_stem=csv_path.stem,
                            paper_name=paper_name,
                            paper_group=paper_group,
                            kernel=kernel,
                            variant=variant,
                            X_train=Xt_train,
                            y_train=yt_train,
                            X_valid=Xt_valid,
                            y_valid=yt_valid,
                            X_test=Xt_test,
                            y_test=yt_test,
                            n_samples=len(frame),
                            n_features=Xt_train.shape[1],
                            args=args,
                        )
                        rows.append(row)
                        tuning_rows.extend(tuning)
                    except Exception as exc:
                        print(f"[error] {exc}", flush=True)
                        rows.append(
                            {
                                **combo,
                                "paper_name": paper_name,
                                "algorithm": VARIANT_LABELS[variant],
                                "status": "error",
                                "error": str(exc),
                            }
                        )
                    if not args.no_flush_each:
                        write_outputs(output_dir, rows, tuning_rows, args)

    write_outputs(output_dir, rows, tuning_rows, args)
    print(f"\nWrote {output_dir / 'comparison_all.csv'}")
    print(f"Wrote {output_dir / 'comparison_deltas_vs_original.csv'}")
    print(f"Wrote {output_dir / 'summary_by_variant.csv'}")
    print(f"Wrote {output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()

