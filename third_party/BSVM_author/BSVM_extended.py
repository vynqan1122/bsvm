# -*- coding: utf-8 -*-
"""Modified, runnable version of the author's four BSVM demo scripts.

This file keeps the upstream Initialsolution -> masterproblem ->
extend_samples flow through ``AuthorExtendedBSVMClassifier`` while exposing
the kernel, c_i function and insertion strategy as command-line options.

For the complete local-dataset protocol use:
    python examples/run_author_ci_comparison.py --search paper --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from roch_bsvm import AuthorExtendedBSVMClassifier, VALID_AUTHOR_VARIANTS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["linear", "poly", "sigmoid", "rbf"], default="linear")
    parser.add_argument("--variant", choices=VALID_AUTHOR_VARIANTS, default="author_original")
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--coef0", type=float, default=0.0)
    parser.add_argument("--n-neighbors", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n_samples = 30
    X_left = rng.normal(size=(n_samples, 2)) * 0.8 + np.array([-1.0, 0.0])
    X_right = rng.normal(size=(n_samples, 2)) * 0.8 + np.array([1.0, 0.0])
    X = np.vstack((X_left, X_right))
    y = np.hstack((-np.ones(n_samples), np.ones(n_samples)))
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=args.seed,
    )

    insertion = "sequential" if args.variant == "author_original" else "binary_tree"
    priority_params = {"n_neighbors": args.n_neighbors}
    gamma: str | float = (
        args.gamma if args.gamma in {"scale", "auto"} else float(args.gamma)
    )
    model = AuthorExtendedBSVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=gamma,
        degree=args.degree,
        coef0=args.coef0,
        class_weight="balanced",
        priority_strategy=args.variant,
        priority_params=priority_params,
        insertion_strategy=insertion,
        random_state=args.seed,
        verbose=1,
    ).fit(
        X_train,
        y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    output = dict(model.fit_summary_)
    output["validation_accuracy"] = float(
        np.mean(model.predict(X_validation) == y_validation)
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
