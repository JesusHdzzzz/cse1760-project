#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split


PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "random_forest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate a random forest on the stroke dataset."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Stroke dataset not found: {args.data}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    df = pd.read_csv(args.data)
    df = df.drop("id", axis=1)
    df = df[df["gender"] != "Other"].copy()

    # Preserved from the original experiment. See part3/README.md before using
    # these results as an unbiased final estimate.
    imputer = SimpleImputer(strategy="median")
    df["bmi"] = imputer.fit_transform(df[["bmi"]]).ravel()
    df_encoded = pd.get_dummies(
        df,
        columns=[
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
        ],
        drop_first=True,
    )

    X = df_encoded.drop("stroke", axis=1)
    y = df_encoded["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y,
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    rf_base = RandomForestClassifier(
        random_state=args.seed,
        class_weight="balanced",
    )
    param_grid = {
        "n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
        "max_depth": [5, 7, 10, 15],
        "min_samples_split": [2, 5, 10],
    }
    grid_search_aupr = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="average_precision",
        cv=2,
        verbose=0,
        n_jobs=-1,
    )
    grid_search_aupr.fit(X_train, y_train)

    best_rf_aupr = grid_search_aupr.best_estimator_
    best_params_aupr = grid_search_aupr.best_params_
    best_score_aupr = grid_search_aupr.best_score_

    y_pred = best_rf_aupr.predict(X_test)
    y_pred_proba = best_rf_aupr.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision_stroke = report["1"]["precision"]
    recall_stroke = report["1"]["recall"]

    print(f"Best AUCPR Score (Cross-Validation): {best_score_aupr:.4f}")
    print(f"Best Hyperparameters: {best_params_aupr}")
    print(f"Test Set ROC AUC: {auc_roc:.4f}")
    print(f"Test Set Average Precision: {auc_pr:.4f}")
    print("\nMetrics for Positive Class (Stroke = 1):")
    print(f"Precision: {precision_stroke * 100:.2f}%")
    print(f"Recall: {recall_stroke * 100:.2f}%")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")
    print(f"\nElapsed time: {time.perf_counter() - start_time:.2f} seconds")

    n_estimators_list = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    test_aucpr_scores = []
    for n_estimators in n_estimators_list:
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=best_params_aupr["max_depth"],
            min_samples_split=best_params_aupr["min_samples_split"],
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_proba = rf.predict_proba(X_test)[:, 1]
        test_aucpr_scores.append(average_precision_score(y_test, y_proba))

    fig, ax = plt.subplots()
    ax.plot(n_estimators_list, test_aucpr_scores, marker="o")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Test Average Precision")
    ax.set_title("Test Average Precision vs Number of Trees")
    ax.grid(True)
    fig.tight_layout()
    output_path = args.output_dir / "average_precision_vs_n_estimators.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
