"""Shared implementation for the two Part 2 MNIST/XGBoost experiments."""

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils_mnist import load_mnist_mat, train_val_split

PCA_VARIANCE = 0.80
TREE_COUNTS = [50, 100, 150, 200]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    display_name: str
    data_path: Path
    output_dir: Path
    cv_folds: int
    search_iterations: int
    parameter_distributions: dict


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    existing = list(path.iterdir()) if path.exists() else []
    if existing and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Choose a new --output-dir "
            "or pass --overwrite explicitly."
        )
    path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_data_path(path: Path, repository_root: Path = PROJECT_ROOT) -> str:
    """Serialize repository data paths portably without leaking host directories."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=PCA_VARIANCE, svd_solver="full")),
            (
                "model",
                XGBClassifier(
                    learning_rate=0.1,
                    objective="multi:softmax",
                    num_class=10,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=1,
                    verbosity=0,
                ),
            ),
        ]
    )


def run_experiment(
    config: ExperimentConfig,
    seed: int,
    train_size: int,
    overwrite: bool,
) -> None:
    start_time = time.perf_counter()
    if not config.data_path.is_file():
        raise FileNotFoundError(f"MNIST dataset not found: {config.data_path}")

    X_all, y_all = load_mnist_mat(config.data_path, "train_fea", "train_gnd")
    X_test, y_test = load_mnist_mat(config.data_path, "test_fea", "test_gnd")
    if set(np.unique(y_all)) != set(range(10)) or set(np.unique(y_test)) != set(range(10)):
        raise ValueError("Both MNIST splits must contain all ten digits after conversion")
    X_train, X_val, y_train, y_val = train_val_split(
        X_all, y_all, train_size=train_size, random_state=seed
    )
    prepare_output_dir(config.output_dir, overwrite)

    print(f"Experiment: {config.display_name}")
    print(f"Train/validation/test rows: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print("Label rule: raw 10 -> digit 0; raw 1..9 remain unchanged")

    search = RandomizedSearchCV(
        estimator=build_pipeline(seed),
        param_distributions=config.parameter_distributions,
        n_iter=config.search_iterations,
        cv=StratifiedKFold(
            n_splits=config.cv_folds, shuffle=True, random_state=seed
        ),
        scoring="accuracy",
        n_jobs=-1,
        random_state=seed,
        verbose=1,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    validation_rows = []
    for n_trees in TREE_COUNTS:
        candidate = clone(search.best_estimator_).set_params(
            model__n_estimators=n_trees
        )
        candidate.fit(X_train, y_train)
        validation_error = 1 - accuracy_score(y_val, candidate.predict(X_val))
        validation_rows.append(
            {"n_estimators": n_trees, "validation_error": validation_error}
        )
        print(f"trees={n_trees:3d} validation error={validation_error:.4f}")

    best_validation = min(validation_rows, key=lambda row: row["validation_error"])
    final_model = clone(search.best_estimator_).set_params(
        model__n_estimators=best_validation["n_estimators"], model__n_jobs=-1
    )
    final_model.fit(X_all, y_all)
    y_pred = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        TREE_COUNTS,
        [row["validation_error"] for row in validation_rows],
        marker="o",
    )
    ax.axvline(best_validation["n_estimators"], linestyle="--")
    ax.set(
        xlabel="Number of trees",
        ylabel="Validation error",
        title=f"{config.display_name}: tree-count selection",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.output_dir / "tree_count_validation_error.png", dpi=300)
    plt.close(fig)

    with (config.output_dir / "model_pipeline.pkl").open("wb") as handle:
        pickle.dump(final_model, handle)
    pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=[f"true_{digit}" for digit in range(10)],
        columns=[f"pred_{digit}" for digit in range(10)],
    ).to_csv(config.output_dir / "test_confusion_matrix.csv")

    metadata = {
        "experiment": config.name,
        "status": "current after corrected digit-label mapping",
        "data_path": portable_data_path(config.data_path),
        "data_sha256": sha256(config.data_path),
        "data_shapes": {"train": list(X_all.shape), "test": list(X_test.shape)},
        "label_mapping": {"raw_10": 0, "raw_1_through_9": "unchanged"},
        "seed": seed,
        "split_sizes": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test),
        },
        "cv_folds": config.cv_folds,
        "search_iterations": config.search_iterations,
        "best_search_params": search.best_params_,
        "best_cv_accuracy": search.best_score_,
        "selected_n_estimators": best_validation["n_estimators"],
        "best_validation_error": best_validation["validation_error"],
        "pca_variance_target": PCA_VARIANCE,
        "pca_components": int(final_model.named_steps["pca"].n_components_),
        "final_test_accuracy": test_accuracy,
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "runtime_seconds": time.perf_counter() - start_time,
        "validation_curve": validation_rows,
    }
    (config.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"Selected trees: {best_validation['n_estimators']}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Saved current artifacts to {config.output_dir}")
