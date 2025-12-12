#!/usr/bin/env python3

import argparse
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import h2o
from h2o.estimators import H2OGradientBoostingEstimator


# ---------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------

def load_and_filter_outliers(
    data_path: str,
    target_col: str = "critical_temp",
    iqr_k: float = 1.5,
    apply_outlier_filter: bool = True,
) -> pd.DataFrame:
    """
    Load superconductivity data and optionally remove outliers in the target
    using IQR: [Q1 - k*IQR, Q3 + k*IQR].
    """
    df = pd.read_csv(data_path)

    if not apply_outlier_filter:
        return df

    y = df[target_col]
    q1 = y.quantile(0.25)
    q3 = y.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_k * iqr
    upper = q3 + iqr_k * iqr

    mask = (y >= lower) & (y <= upper)
    filtered_df = df.loc[mask].reset_index(drop=True)

    print(f"[INFO] Outlier filtering on '{target_col}':")
    print(f"       Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
    print(f"       Keeping {filtered_df.shape[0]} / {df.shape[0]} rows "
          f"({100.0 * filtered_df.shape[0] / df.shape[0]:.2f}% retained)")

    return filtered_df


def train_test_split_shared(
    df: pd.DataFrame,
    target_col: str = "critical_temp",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Shared train/test split for all models (H2O + sklearn).
    Returns:
        X_train, X_test, y_train, y_test (pandas)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    print(f"[INFO] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def to_h2o_frames(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_col: str = "critical_temp",
):
    """
    Convert train/test (pandas) to H2OFrames.
    """
    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()
    test_df[target_col] = y_test.values

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    x_cols = [c for c in train_h2o.columns if c != target_col]
    y_col = target_col

    return train_h2o, test_h2o, x_cols, y_col


# ---------------------------------------------------------------------
# H2O GBM helpers
# ---------------------------------------------------------------------

def get_h2o_cv_and_test_metrics(model, test_frame) -> Dict[str, float]:
    """
    Extract CV RMSE/MAE and Test RMSE/MAE for an H2O model.
    Uses the model's metric accessors with xval=True for CV.
    """
    # Cross-validation metrics
    cv_rmse = model.rmse(xval=True)
    try:
        cv_mae = model.mae(xval=True)
    except Exception:
        cv_mae = np.nan

    # Test metrics
    perf_test = model.model_performance(test_frame)
    test_rmse = perf_test.rmse()
    try:
        test_mae = perf_test.mae()
    except Exception:
        test_mae = np.nan

    return {
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }


def h2o_tree_model_complexity(model) -> Dict[str, Any]:
    """
    Approximate complexity for tree-based H2O models (GBM).
    Uses model_summary to fetch global counts.
    """
    try:
        summary = model._model_json["output"]["model_summary"].as_data_frame()
        n_trees = int(summary.loc[0, "number_of_trees"])
        n_internal = int(summary.loc[0, "number_of_internal_trees"])
        max_depth = float(summary.loc[0, "max_depth"])
        mean_depth = float(summary.loc[0, "mean_depth"])
        mean_leaves = float(summary.loc[0, "mean_leaves"])
        approx_params = n_internal + int(mean_leaves * n_trees)
        return {
            "n_trees": n_trees,
            "approx_params": approx_params,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "mean_leaves": mean_leaves,
        }
    except Exception as e:
        print(f"[WARN] Could not compute tree complexity: {e}")
        return {
            "n_trees": np.nan,
            "approx_params": np.nan,
            "max_depth": np.nan,
            "mean_depth": np.nan,
            "mean_leaves": np.nan,
        }


# ---------------------------------------------------------------------
# Base GBM + sweep configs
# ---------------------------------------------------------------------

def run_h2o_gbm(train, test, x_cols, y_col, seed=42) -> Dict[str, Any]:
    """
    Train base GBM with 5-fold CV and return metrics + complexity.
    """
    print("[INFO] Training H2O GBM (base config)...")
    start = time.perf_counter()

    # Base GBM hyperparameters
    base_params = {
        "ntrees": 500,
        "max_depth": 8,
        "learn_rate": 0.05,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 5,
    }

    gbm = H2OGradientBoostingEstimator(
        ntrees=base_params["ntrees"],
        max_depth=base_params["max_depth"],
        learn_rate=base_params["learn_rate"],
        sample_rate=base_params["sample_rate"],
        col_sample_rate=base_params["col_sample_rate"],
        min_rows=base_params["min_rows"],
        nfolds=5,
        seed=seed,
        keep_cross_validation_models=True,
        # early stopping
        stopping_rounds=0,
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        score_tree_interval=5,
    )
    gbm.train(x=x_cols, y=y_col, training_frame=train)

    elapsed = time.perf_counter() - start
    metrics = get_h2o_cv_and_test_metrics(gbm, test)
    complexity = h2o_tree_model_complexity(gbm)

    result = {
        "model_name": "H2O_GBM_Base",
        "algorithm": "GBM",
        "train_time_sec": elapsed,
        "cv_rmse": metrics["cv_rmse"],
        "cv_mae": metrics["cv_mae"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "n_params": complexity["approx_params"],
        "n_trees": complexity["n_trees"],
        "max_depth": complexity["max_depth"],
        "mean_depth": complexity["mean_depth"],
        # include hyperparams explicitly for summary table
        "learn_rate": base_params["learn_rate"],
        "sample_rate": base_params["sample_rate"],
        "col_sample_rate": base_params["col_sample_rate"],
        "min_rows": base_params["min_rows"],
    }
    return result

# These are the new tests you added:
GBM_SWEEP_CONFIGS = [

    # ntrees 1200
    {
        "name": "GBM_n1200_d12",
        "ntrees": 1200,
        "max_depth": 12,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1200_d11",
        "ntrees": 1200,
        "max_depth": 11,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1200_d10",
        "ntrees": 1200,
        "max_depth": 10,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1200_d9",
        "ntrees": 1200,
        "max_depth": 9,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    # ntrees 1100
    {
        "name": "GBM_n1100_d12",
        "ntrees": 1100,
        "max_depth": 12,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1100_d11",
        "ntrees": 1100,
        "max_depth": 11,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1100_d10",
        "ntrees": 1100,
        "max_depth": 10,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    {
        "name": "GBM_n1100_d9",
        "ntrees": 1100,
        "max_depth": 9,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    # ntrees 1000
    #{
    #    "name": "GBM_n900_d12",
    #    "ntrees": 1000,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d11",
    #    "ntrees": 1000,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d10",
    #    "ntrees": 1000,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d9",
    #    "ntrees": 1000,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    # ntrees 900
    #{
    #    "name": "GBM_n900_d12",
    #    "ntrees": 900,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d11",
    #    "ntrees": 900,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d10",
    #    "ntrees": 900,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n900_d9",
    #    "ntrees": 900,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    # ntrees 800
    #{
    #    "name": "GBM_n800_d12",
    #    "ntrees": 800,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n800_d11",
    #    "ntrees": 800,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n800_d10",
    #    "ntrees": 800,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n800_d9",
    #    "ntrees": 800,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
#
    ## ntrees 700
    #{
    #    "name": "GBM_n700_d12",
    #    "ntrees": 700,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n700_d11",
    #    "ntrees": 700,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n700_d10",
    #    "ntrees": 700,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n700_d9",
    #    "ntrees": 700,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
#
    ## ntrees 600
    #{
    #    "name": "GBM_n600_d12",
    #    "ntrees": 600,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n600_d11",
    #    "ntrees": 600,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n600_d10",
    #    "ntrees": 600,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n600_d9",
    #    "ntrees": 600,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    ## ntrees 500
    #{
    #    "name": "GBM_n500_d12",
    #    "ntrees": 500,
    #    "max_depth": 12,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n500_d11",
    #    "ntrees": 500,
    #    "max_depth": 11,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n500_d10",
    #    "ntrees": 500,
    #    "max_depth": 10,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
    #{
    #    "name": "GBM_n500_d9",
    #    "ntrees": 500,
    #    "max_depth": 9,
    #    "learn_rate": 0.03,
    #    "sample_rate": 0.8,
    #    "col_sample_rate": 0.8,
    #    "min_rows": 10,
    #},
]

def run_gbm_sweep(train, test, x_cols, y_col, seed=42) -> List[Dict[str, Any]]:
    """
    Run a sweep of manual GBM configurations you defined in GBM_SWEEP_CONFIGS.
    Returns a list of result dicts.
    """
    sweep_results: List[Dict[str, Any]] = []

    for cfg in GBM_SWEEP_CONFIGS:
        print(f"[INFO] GBM Sweep: Training config {cfg['name']}...")
        start = time.perf_counter()

        gbm = H2OGradientBoostingEstimator(
            ntrees=cfg["ntrees"],
            max_depth=cfg["max_depth"],
            learn_rate=cfg["learn_rate"],
            sample_rate=cfg["sample_rate"],
            col_sample_rate=cfg["col_sample_rate"],
            min_rows=cfg["min_rows"],
            nfolds=5,
            seed=seed,
            # early stopping
            stopping_rounds=0,
            stopping_metric="RMSE",
            stopping_tolerance=0.001,
            score_tree_interval=5,
        )
        gbm.train(x=x_cols, y=y_col, training_frame=train)

        elapsed = time.perf_counter() - start
        metrics = get_h2o_cv_and_test_metrics(gbm, test)
        complexity = h2o_tree_model_complexity(gbm)

        sweep_results.append({
            "model_name": cfg["name"],
            "algorithm": "GBM_sweep",
            "train_time_sec": elapsed,
            "cv_rmse": metrics["cv_rmse"],
            "cv_mae": metrics["cv_mae"],
            "test_rmse": metrics["test_rmse"],
            "test_mae": metrics["test_mae"],
            "n_params": complexity["approx_params"],
            "n_trees": complexity["n_trees"],
            "max_depth": cfg["max_depth"],
            "mean_depth": complexity["mean_depth"],
            "learn_rate": cfg["learn_rate"],
            "sample_rate": cfg["sample_rate"],
            "col_sample_rate": cfg["col_sample_rate"],
            "min_rows": cfg["min_rows"],
        })

    return sweep_results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GBM-only Superconductivity Experiments with Manual Sweep"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../Part3/images/superconductivty-data/train.csv",
        help="Path to superconductivity CSV (must include 'critical_temp').",
    )
    parser.add_argument(
        "--no-outlier-filter",
        action="store_true",
        help="Disable IQR-based outlier removal on critical_temp.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results_superconductivity_gbm_sweep.csv",
        help="Where to save the results table.",
    )
    parser.add_argument(
        "--run-gbm-sweep",
        action="store_true",
        help="Run additional GBM hyperparameter sweep experiments.",
    )
    args = parser.parse_args()

    # 1. Load data & filter outliers
    df = load_and_filter_outliers(
        data_path=args.data_path,
        target_col="critical_temp",
        iqr_k=1.5,
        apply_outlier_filter=not args.no_outlier_filter,
    )

    # 2. Shared train/test split
    X_train, X_test, y_train, y_test = train_test_split_shared(
        df,
        target_col="critical_temp",
        test_size=0.2,
        random_state=42,
    )

    # 3. Init H2O and create H2OFrames
    print("[INFO] Initializing H2O...")
    h2o.init()
    h2o.show_progress()

    train_h2o, test_h2o, x_cols, y_col = to_h2o_frames(
        X_train, X_test, y_train, y_test, target_col="critical_temp"
    )

    results: List[Dict[str, Any]] = []

    # Base GBM
    results.append(run_h2o_gbm(train_h2o, test_h2o, x_cols, y_col))

    # Manual GBM sweep
    if args.run_gbm_sweep:
        sweep_results = run_gbm_sweep(train_h2o, test_h2o, x_cols, y_col)
        results.extend(sweep_results)

    # Build summary table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_rmse").reset_index(drop=True)

    # Make a summary with column names similar to your earlier GBM table
    summary_df = results_df.rename(
        columns={
            "model_name": "experiment",
            "cv_rmse": "valid_RMSE",
            "cv_mae": "valid_MAE",
        }
    )

    cols_to_show = [
        "experiment",
        "valid_RMSE",
        "valid_MAE",
        "test_rmse",
        "test_mae",
        "train_time_sec",
        "n_trees",
        "max_depth",
        "learn_rate",
        "sample_rate",
        "col_sample_rate",
        "min_rows",
        "n_params",
    ]

    print("\n===== Building final summary table (GBM base + sweep) =====")
    print(summary_df[cols_to_show])
    print("==========================================================\n")

    summary_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Saved results to {args.output_csv}")

    # Shutdown H2O (optional)
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()