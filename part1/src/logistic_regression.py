#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import DATA_PATH, encode_binary_labels, filter_digits, load_mnist, split_data

PART1_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PART1_DIR / "outputs" / "logistic_regression"
C_VALUES = [0.001, 0.01, 0.1, 1, 10, 100, 1000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate logistic regression for MNIST digits 5 vs 6."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def build_model(c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    X, y = load_mnist(args.data_path)
    X, y = filter_digits(X, y, digits=(5, 6))
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        train_size=args.train_size,
        val_size=args.validation_size,
        test_size=args.test_size,
        random_state=args.seed,
    )
    y_train = encode_binary_labels(y_train)
    y_val = encode_binary_labels(y_val)
    y_test = encode_binary_labels(y_test)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows = []
    for c_value in C_VALUES:
        model = build_model(c_value, args.seed)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        row = {
            "C": c_value,
            "train_accuracy": model.score(X_train, y_train),
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "validation_accuracy": model.score(X_val, y_val),
        }
        rows.append(row)
        print(
            f"C={c_value:7.3f} train={row['train_accuracy']:.4f} "
            f"CV={row['cv_accuracy_mean']:.4f} +/- {row['cv_accuracy_std']:.4f} "
            f"validation={row['validation_accuracy']:.4f}"
        )

    best = max(rows, key=lambda row: row["cv_accuracy_mean"])
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    final_model = build_model(best["C"], args.seed)
    final_model.fit(X_combined, y_combined)
    test_accuracy = accuracy_score(y_test, final_model.predict(X_test))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(
        C_VALUES,
        [1 - row["train_accuracy"] for row in rows],
        "o-",
        label="Training error",
    )
    ax.semilogx(
        C_VALUES,
        [1 - row["validation_accuracy"] for row in rows],
        "s-",
        label="Validation error",
    )
    ax.axvline(best["C"], linestyle="--", label=f"CV-selected C={best['C']}")
    ax.set(
        xlabel="Regularization parameter C",
        ylabel="Error rate",
        title="Logistic regression model selection",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "training_validation_error_by_c.png", dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)

    results = {
        "seed": args.seed,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "selection_metric": "five-fold cross-validation accuracy on training split",
        "selected_C": best["C"],
        "selected_cv_accuracy": best["cv_accuracy_mean"],
        "selected_validation_accuracy_before_refit": best["validation_accuracy"],
        "final_fit_rows": len(y_combined),
        "final_test_accuracy": test_accuracy,
        "candidates": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (args.output_dir / "model_pipeline.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    print(f"Selected C: {best['C']}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("Candidate metrics and final test metrics describe different fitted models.")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
