#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_utils import DATA_PATH, encode_binary_labels, filter_digits, load_mnist, split_data

PART1_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PART1_DIR / "outputs" / "random_forest"
TREE_COUNTS = [10, 50, 100, 200, 500]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a random forest for MNIST digits 5 vs 6."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


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
    for n_trees in TREE_COUNTS:
        model = RandomForestClassifier(
            n_estimators=n_trees, random_state=args.seed, n_jobs=-1
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        model.fit(X_train, y_train)
        row = {
            "n_estimators": n_trees,
            "train_accuracy": model.score(X_train, y_train),
            "cv_accuracy_mean": scores.mean(),
            "cv_accuracy_std": scores.std(),
            "validation_accuracy": model.score(X_val, y_val),
        }
        rows.append(row)
        print(
            f"trees={n_trees:3d} train={row['train_accuracy']:.4f} "
            f"CV={row['cv_accuracy_mean']:.4f} +/- {row['cv_accuracy_std']:.4f} "
            f"validation={row['validation_accuracy']:.4f}"
        )

    best = max(rows, key=lambda row: row["cv_accuracy_mean"])
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    final_model = RandomForestClassifier(
        n_estimators=best["n_estimators"], random_state=args.seed, n_jobs=-1
    )
    final_model.fit(X_combined, y_combined)
    test_accuracy = accuracy_score(y_test, final_model.predict(X_test))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        TREE_COUNTS,
        [1 - row["train_accuracy"] for row in rows],
        "o-",
        label="Training error",
    )
    ax.plot(
        TREE_COUNTS,
        [1 - row["validation_accuracy"] for row in rows],
        "s-",
        label="Validation error",
    )
    ax.axvline(
        best["n_estimators"],
        linestyle="--",
        label=f"CV-selected trees={best['n_estimators']}",
    )
    ax.set(
        xlabel="Number of trees",
        ylabel="Error rate",
        title="Random forest model selection",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "model_selection_error.png", dpi=300)
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
        "selected_n_estimators": best["n_estimators"],
        "selected_cv_accuracy": best["cv_accuracy_mean"],
        "selected_validation_accuracy_before_refit": best["validation_accuracy"],
        "final_fit_rows": len(y_combined),
        "final_test_accuracy": test_accuracy,
        "candidates": rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (args.output_dir / "model.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    print(f"Selected trees: {best['n_estimators']}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print("Candidate metrics and final test metrics describe different fitted models.")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
