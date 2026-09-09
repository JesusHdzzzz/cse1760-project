#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from h2o_utils import init_h2o, shutdown_h2o_if_owned
from stroke_utils import load_stroke_data, stratified_stroke_split

PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "stroke_h2o_gbm"
BMI_MISSING_VALUE_POLICY = "native H2O GBM missing-value handling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate an H2O GBM on the stroke dataset."
    )
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-mem-size", default="8G")
    parser.add_argument("--keep-h2o-cluster", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def best_f2_threshold(y_true, scores):
    best = (-1.0, 0.5)
    for threshold in np.linspace(0.01, 0.99, 200):
        tn, fp, fn, tp = confusion_matrix(
            y_true, scores >= threshold, labels=[0, 1]
        ).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = 4 * precision + recall
        f2 = 5 * precision * recall / denominator if denominator else 0.0
        if f2 > best[0]:
            best = (f2, float(threshold))
    return best


def save_figure(fig, path: Path, show: bool) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def split_raw_stroke_data(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the outer split without imputing H2O predictor values."""
    return stratified_stroke_split(df, test_size, seed)


def select_model_and_f2_threshold(grid, train, target: str):
    """Select from training CV results and its out-of-fold predictions only."""
    best_model = grid.get_grid(sort_by="aucpr", decreasing=True).models[0]
    cv_scores = best_model.cross_validation_holdout_predictions().as_data_frame()[
        "p1"
    ].to_numpy()
    cv_true = train[target].as_data_frame()[target].astype(int).to_numpy()
    cv_f2, threshold = best_f2_threshold(cv_true, cv_scores)
    return best_model, cv_f2, threshold


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")

    start = time.perf_counter()
    df = load_stroke_data(args.data_path)
    train_df, test_df = split_raw_stroke_data(df, args.test_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    owned_cluster = init_h2o(args.max_mem_size)
    try:
        train = h2o.H2OFrame(train_df)
        target = "stroke"
        features = [column for column in train.columns if column not in {"id", target}]
        train[target] = train[target].asfactor()

        hyper_params = {
            "max_depth": [4, 6, 8],
            "learn_rate": [0.01, 0.03, 0.05],
            "ntrees": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sample_rate": [0.7, 0.8, 0.9],
            "col_sample_rate": [0.7, 0.8, 0.9],
            "min_rows": [2, 3, 4],
        }
        base = H2OGradientBoostingEstimator(
            seed=args.seed,
            balance_classes=True,
            class_sampling_factors=[1.0, 11.0],
            nfolds=args.cv_folds,
            keep_cross_validation_predictions=True,
            fold_assignment="Stratified",
        )
        grid = H2OGridSearch(
            model=base,
            hyper_params=hyper_params,
            search_criteria={"strategy": "Cartesian"},
        )
        grid.train(x=features, y=target, training_frame=train)

        rows = [
            {
                "ntrees": model.actual_params["ntrees"],
                "cv_aucpr": model.aucpr(xval=True),
            }
            for model in grid.models
        ]
        curve = (
            pd.DataFrame(rows)
            .groupby("ntrees", as_index=False)["cv_aucpr"]
            .max()
        )
        curve.to_csv(args.output_dir / "tree_count_cv_aucpr.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(curve["ntrees"], curve["cv_aucpr"], marker="o")
        ax.set(
            xlabel="Number of trees",
            ylabel="Best H2O cross-validation AUCPR",
            title="Stroke H2O GBM model selection",
        )
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_figure(
            fig, args.output_dir / "tree_count_cv_aucpr.png", args.show
        )

        best_model, cv_f2, threshold = select_model_and_f2_threshold(
            grid, train, target
        )

        # Final test access occurs only after model and decision threshold selection.
        test = h2o.H2OFrame(test_df)
        test[target] = test[target].asfactor()
        test_scores = best_model.predict(test).as_data_frame()["p1"].to_numpy()
        y_test = test[target].as_data_frame()[target].astype(int).to_numpy()
        y_pred = (test_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

        precision_values, recall_values, _ = precision_recall_curve(y_test, test_scores)
        average_precision = average_precision_score(y_test, test_scores)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall_values, precision_values, label=f"Average precision={average_precision:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", title="Stroke H2O GBM precision-recall curve")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, args.output_dir / "precision_recall_curve.png", args.show)

        fpr, tpr, _ = roc_curve(y_test, test_scores)
        roc_auc = roc_auc_score(y_test, test_scores)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Stroke H2O GBM ROC curve")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, args.output_dir / "roc_curve.png", args.show)

        metadata = {
            "seed": args.seed,
            "split": (
                "stratified train/test split with raw predictor missingness retained"
            ),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "bmi_missing_value_policy": BMI_MISSING_VALUE_POLICY,
            "cv_folds": args.cv_folds,
            "selection_metric": "H2O cross-validation AUCPR",
            "cv_selected_f2_threshold": threshold,
            "cv_f2_at_selected_threshold": cv_f2,
            "best_model_params": {
                key: best_model.actual_params[key] for key in hyper_params
            },
            "test_average_precision": average_precision,
            "test_roc_auc": roc_auc,
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "runtime_seconds": time.perf_counter() - start,
        }
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        model_path = h2o.save_model(best_model, path=str(args.output_dir), force=True)
        print(f"Final test average precision: {average_precision:.4f}")
        print(f"Final test ROC AUC: {roc_auc:.4f}")
        print(f"Saved model: {model_path}")
    finally:
        shutdown_h2o_if_owned(owned_cluster, args.keep_h2o_cluster)


if __name__ == "__main__":
    main()
