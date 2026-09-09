#!/usr/bin/env python3

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from stroke_utils import load_stroke_data

PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "stroke_random_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a random forest on the stroke dataset."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def build_pipeline(df: pd.DataFrame, seed: int) -> Pipeline:
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in df.columns if column not in categorical]
    preprocess = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=seed, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

    start = time.perf_counter()
    df = load_stroke_data(args.data_path)
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv = StratifiedKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed
    )
    search = GridSearchCV(
        estimator=build_pipeline(X_train, args.seed),
        param_grid={
            "classifier__n_estimators": [
                50,
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500,
                550,
                600,
            ],
            "classifier__max_depth": [5, 7, 10, 15],
            "classifier__min_samples_split": [2, 5, 10],
        },
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_)
    curve = (
        cv_results.groupby("param_classifier__n_estimators", as_index=False)[
            "mean_test_score"
        ]
        .max()
        .rename(
            columns={
                "param_classifier__n_estimators": "n_estimators",
                "mean_test_score": "best_cv_average_precision",
            }
        )
    )
    curve.to_csv(args.output_dir / "tree_count_cv_average_precision.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["n_estimators"], curve["best_cv_average_precision"], marker="o")
    ax.set(
        xlabel="Number of trees",
        ylabel="Best cross-validation average precision",
        title="Stroke random forest model selection",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "tree_count_cv_average_precision.png", dpi=300)
    if args.show:
        plt.show()
    plt.close(fig)

    # The test set is accessed once, after all model and threshold choices are fixed.
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "seed": args.seed,
        "split": "stratified 80/20 train/test",
        "cv_folds": args.cv_folds,
        "selection_metric": "cross-validation average precision",
        "best_cv_average_precision": search.best_score_,
        "best_params": search.best_params_,
        "test_average_precision": average_precision_score(y_test, y_score),
        "test_roc_auc": roc_auc_score(y_test, y_score),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_classification_report": report,
        "runtime_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    with (args.output_dir / "model_pipeline.pkl").open("wb") as handle:
        pickle.dump(model, handle)

    print(f"Best CV average precision: {search.best_score_:.4f}")
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Final test average precision: {metrics['test_average_precision']:.4f}")
    print(f"Final test ROC AUC: {metrics['test_roc_auc']:.4f}")
    print(f"Saved current artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
