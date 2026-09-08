#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


PART3_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PART3_DIR / "data" / "healthcare-dataset-stroke-data.csv"
OUTPUT_DIR = PART3_DIR / "outputs" / "h2o_stroke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and evaluate an H2O GBM on the stroke dataset."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-mem-size", default="8G")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def save_figure(fig, path: Path, show: bool) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Stroke dataset not found: {args.data}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    df = pd.read_csv(args.data)

    # Preserved from the original experiment. See part3/README.md before using
    # these results as an unbiased final estimate.
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)
    df = df[df["gender"] != "Other"].copy()

    h2o.init(max_mem_size=args.max_mem_size)
    try:
        h2o_df = h2o.H2OFrame(df)
        target = "stroke"
        features = [column for column in h2o_df.columns if column not in {"id", target}]
        h2o_df[target] = h2o_df[target].asfactor()

        train, test = h2o_df.split_frame(ratios=[0.8], seed=args.seed)

        print("\n--- Class Distribution in Splits ---")
        train_counts = train[target].table().as_data_frame()
        test_counts = test[target].table().as_data_frame()
        print("\nTraining Set:")
        print(
            f"No Stroke: {int(train_counts.iloc[0, 1])}, "
            f"Stroke: {int(train_counts.iloc[1, 1])}"
        )
        print("\nTest Set:")
        print(
            f"No Stroke: {int(test_counts.iloc[0, 1])}, "
            f"Stroke: {int(test_counts.iloc[1, 1])}"
        )
        total_no_stroke = int(train_counts.iloc[0, 1]) + int(test_counts.iloc[0, 1])
        total_stroke = int(train_counts.iloc[1, 1]) + int(test_counts.iloc[1, 1])
        print(f"Totals: No Stroke: {total_no_stroke}, Stroke: {total_stroke}")

        hyper_params = {
            "max_depth": [4, 6, 8],
            "learn_rate": [0.01, 0.03, 0.05],
            "ntrees": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "sample_rate": [0.7, 0.8, 0.9],
            "col_sample_rate": [0.7, 0.8, 0.9],
            "min_rows": [2, 3, 4],
        }
        gbm_base = H2OGradientBoostingEstimator(
            seed=args.seed,
            balance_classes=True,
            class_sampling_factors=[1.0, 11.0],
            nfolds=2,
            keep_cross_validation_predictions=True,
            fold_assignment="Stratified",
        )
        grid = H2OGridSearch(
            model=gbm_base,
            hyper_params=hyper_params,
            search_criteria={"strategy": "Cartesian"},
        )
        grid.train(x=features, y=target, training_frame=train)

        rows = []
        for model in grid.models:
            ntrees = model.actual_params["ntrees"]
            test_aucpr = model.model_performance(test_data=test).aucpr()
            rows.append((ntrees, test_aucpr))

        best_by_ntrees = {}
        for ntrees, aucpr_value in rows:
            best_by_ntrees[ntrees] = max(
                best_by_ntrees.get(ntrees, -1),
                aucpr_value,
            )

        ntree_values = sorted(best_by_ntrees)
        aucpr_values = [best_by_ntrees[value] for value in ntree_values]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ntree_values, aucpr_values, marker="o", linewidth=2)
        ax.set_xlabel("Number of Trees (ntrees)")
        ax.set_ylabel("Test AUCPR")
        ax.set_title("Test AUCPR vs Number of Trees")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(
            fig,
            args.output_dir / "test_aucpr_vs_ntrees.png",
            args.show,
        )

        grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)
        print("\n--- Top Models from Grid Search (Sorted by CV AUCPR) ---")
        print(grid_perf)
        best_gbm = grid_perf.models[0]

        cv_pred = best_gbm.cross_validation_holdout_predictions()
        cv_scores = cv_pred.as_data_frame()["p1"].values
        cv_true = train[target].as_data_frame()[target].values

        thresholds = np.linspace(0.01, 0.99, 200)
        best_f2 = -1.0
        best_threshold = None
        for threshold in thresholds:
            y_pred = (cv_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                cv_true,
                y_pred,
                labels=[0, 1],
            ).ravel()
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            denominator = 4 * precision + recall
            f2 = 5 * precision * recall / denominator if denominator else 0.0
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold

        test_scores = best_gbm.predict(test).as_data_frame()["p1"].values
        y_test = test[target].as_data_frame()[target].values
        y_test_pred = (test_scores >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_test_pred,
            labels=[0, 1],
        ).ravel()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        print("\n--- Best F2 Threshold (Cross-Validation) ---")
        print(f"Threshold: {best_threshold:.3f}")
        print(f"Cross-validation F2 Score: {best_f2:.4f}")
        print("\n--- Test Confusion Matrix at CV-Selected F2 Threshold ---")
        print(f"TN: {tn}, FP: {fp}")
        print(f"FN: {fn}, TP: {tp}")
        print("\n--- Test Metrics ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        precision_values, recall_values, _ = precision_recall_curve(y_test, test_scores)
        average_precision = average_precision_score(y_test, test_scores)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            recall_values,
            precision_values,
            linewidth=2,
            label=f"Average precision = {average_precision:.3f}",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve (Best GBM)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, args.output_dir / "precision_recall_curve.png", args.show)

        fpr, tpr, _ = roc_curve(y_test, test_scores)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve (Best GBM)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        save_figure(fig, args.output_dir / "roc_curve.png", args.show)

        print(f"Best model test average precision: {average_precision:.4f}")
        print(f"Best model test ROC AUC: {roc_auc:.4f}")
        print("\n--- Best Model Hyperparameters ---")
        print(
            f"Max Depth: {best_gbm.actual_params['max_depth']}, "
            f"Learning Rate: {best_gbm.actual_params['learn_rate']}, "
            f"Number of Trees: {best_gbm.actual_params['ntrees']}, "
            f"Sample Rate: {best_gbm.actual_params['sample_rate']}, "
            f"Col Sample Rate: {best_gbm.actual_params['col_sample_rate']}, "
            f"Min Rows: {best_gbm.actual_params['min_rows']}"
        )
        print(f"\nElapsed time: {time.perf_counter() - start_time:.2f} seconds")
    finally:
        h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
