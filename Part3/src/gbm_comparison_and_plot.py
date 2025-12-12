#!/usr/bin/env python3

import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import h2o
from h2o import H2OFrame
from h2o.estimators import H2OGradientBoostingEstimator


def plot_model_diagnostics(experiment_name: str, model, test_h2o, y_col: str):
    """
    Create:
      1) Predicted vs Actual scatter
      2) Residual history (index vs residual)
      3) Residuals vs Predicted scatter
    for a given model and test frame.
    """
    # Predictions and true values
    pred_h2o = model.predict(test_h2o)
    y_true_pd = test_h2o[y_col].as_data_frame(use_pandas=True)
    y_pred_pd = pred_h2o.as_data_frame(use_pandas=True)

    df_eval = pd.DataFrame({
        "y_true": y_true_pd[y_col],
        "y_pred": y_pred_pd["predict"],
    })
    df_eval["residual"] = df_eval["y_true"] - df_eval["y_pred"]

    # For sanity check
    rmse = ((df_eval["residual"] ** 2).mean()) ** 0.5
    print(f"[{experiment_name}] RMSE from residuals: {rmse:.4f}")

    # 1) Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(df_eval["y_true"], df_eval["y_pred"], alpha=0.4)
    plt.xlabel("Actual critical_temp")
    plt.ylabel("Predicted critical_temp")
    plt.title(f"Predicted vs Actual ({experiment_name})")
    # 45-degree line
    min_val = min(df_eval["y_true"].min(), df_eval["y_pred"].min())
    max_val = max(df_eval["y_true"].max(), df_eval["y_pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.tight_layout()
    plt.savefig(f"new_output/{experiment_name}_pred_vs_actual.png", dpi=200)

    # 2) Residual history
    plt.figure(figsize=(10, 4))
    plt.plot(df_eval["residual"].values)
    plt.xlabel("Test sample index")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals over test set ({experiment_name})")
    plt.tight_layout()
    plt.savefig(f"new_output/{experiment_name}_residual_history.png", dpi=200)

    # 3) Residuals vs Predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(df_eval["y_pred"], df_eval["residual"], alpha=0.4)
    plt.axhline(0)
    plt.xlabel("Predicted critical_temp")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted ({experiment_name})")
    plt.tight_layout()
    plt.savefig(f"new_output/{experiment_name}_residual_scatter.png", dpi=200)

    print(f"[{experiment_name}] Saved diagnostic plots.")


# ---------------------------------------------------------------------
# 1. Data loading & preprocessing (your original functions)
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

    n_total = df.shape[0]
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    print("\n[INFO] Train/Test split (after global outlier filter)")
    print(f"       Total rows: {n_total}")
    print(f"       Train size: {n_train} ({100.0 * n_train / n_total:.2f}%)")
    print(f"       Test size:  {n_test} ({100.0 * n_test / n_total:.2f}%)")

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
# 2. Helpers: metrics & complexity (your originals)
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
    Uses model_summary to fetch global counts when available.
    Always tries to return a valid n_trees.
    """
    try:
        # H2O GBM usually stores things under "output"
        output = model._model_json.get("output", model._model_json.get("new_output"))
        summary = output["model_summary"].as_data_frame()

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
        # Fallback: still try to recover n_trees directly from params
        print(f"[WARN] Could not compute full tree complexity: {e}")

        try:
            n_trees = int(model.params["ntrees"]["actual"])
        except Exception:
            n_trees = np.nan

        return {
            "n_trees": n_trees,
            "approx_params": np.nan,
            "max_depth": np.nan,
            "mean_depth": np.nan,
            "mean_leaves": np.nan,
        }

# ---------------------------------------------------------------------
# 3. Base GBM + sweep configs (matching your scripts)
# ---------------------------------------------------------------------

def run_h2o_gbm_base(train, test, x_cols, y_col, seed=42) -> Dict[str, Any]:
    """
    Train base GBM with 5-fold CV and return metrics + complexity.
    """
    print("[INFO] Training H2O GBM (base config)...")
    start = time.perf_counter()

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
        # early stopping on CV (same as your original base)
        stopping_rounds=10,
        stopping_metric="RMSE",
        stopping_tolerance=0.001,
        score_tree_interval=5,
    )
    gbm.train(x=x_cols, y=y_col, training_frame=train)

    elapsed = time.perf_counter() - start
    metrics = get_h2o_cv_and_test_metrics(gbm, test)
    complexity = h2o_tree_model_complexity(gbm)

    result = {
        "experiment": "H2O_GBM_Base",
        "train_time_sec": elapsed,
        "valid_RMSE": metrics["cv_rmse"],
        "valid_MAE": metrics["cv_mae"],
        "test_RMSE": metrics["test_rmse"],
        "test_MAE": metrics["test_mae"],
        "n_params": complexity["approx_params"],
        "n_trees": complexity["n_trees"],
        "max_depth": complexity["max_depth"],
        "mean_depth": complexity["mean_depth"],
        "learn_rate": base_params["learn_rate"],
        "sample_rate": base_params["sample_rate"],
        "col_sample_rate": base_params["col_sample_rate"],
        "min_rows": base_params["min_rows"],
        "model": gbm,
    }
    return result


GBM_SWEEP_CONFIGS = [
    # 1100-tree, depth-10 config (best one)
    {
        "name": "GBM_n1100_d10",
        "ntrees": 1100,
        "max_depth": 10,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    # Base model (depth=12)
    {
        "name": "Base_n500_d12",
        "ntrees": 500,
        "max_depth": 8,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
]


def run_gbm_sweep(train, test, x_cols, y_col, seed=42) -> List[Dict[str, Any]]:
    """
    Run a sweep of manual GBM configurations (from GBM_SWEEP_CONFIGS).
    Uses 5-fold CV, no early stopping (matches your final sweep version).
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
            keep_cross_validation_models=False,
        )
        gbm.train(x=x_cols, y=y_col, training_frame=train)

        elapsed = time.perf_counter() - start
        metrics = get_h2o_cv_and_test_metrics(gbm, test)
        complexity = h2o_tree_model_complexity(gbm)

        sweep_results.append({
            "experiment": cfg["name"],
            "train_time_sec": elapsed,
            "valid_RMSE": metrics["cv_rmse"],
            "valid_MAE": metrics["cv_mae"],
            "test_RMSE": metrics["test_rmse"],
            "test_MAE": metrics["test_mae"],
            "n_params": complexity["approx_params"],
            "n_trees": complexity["n_trees"],
            "max_depth": cfg["max_depth"],
            "mean_depth": complexity["mean_depth"],
            "learn_rate": cfg["learn_rate"],
            "sample_rate": cfg["sample_rate"],
            "col_sample_rate": cfg["col_sample_rate"],
            "min_rows": cfg["min_rows"],
            "model": gbm,
        })

    return sweep_results


# ---------------------------------------------------------------------
# 4. Extra experiment: "No Target Outliers" (train only)
# ---------------------------------------------------------------------

def run_gbm_no_target_outliers_train_only(
    name: str,
    strong_params: Dict[str, Any],
    train_h2o: H2OFrame,
    test_h2o: H2OFrame,
    x_cols: List[str],
    y_col: str,
) -> Dict[str, Any]:
    """
    Reproduce the idea from your 'No Target Outliers' script:
    - Convert train frame to pandas
    - Compute IQR on train target only
    - Filter train only (valid/test unchanged)
    - Train GBM with strong_params
    """
    print("\n===== GBM with target outliers removed (train only, IQR filter) =====")
    train_pd = train_h2o.as_data_frame(use_pandas=True)

    y = y_col
    q1 = train_pd[y].quantile(0.25)
    q3 = train_pd[y].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    print(f"Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}")
    print(f"Lower bound={lower:.3f}, Upper bound={upper:.3f}")

    train_no_outliers_pd = train_pd[(train_pd[y] >= lower) & (train_pd[y] <= upper)]
    print("Original train rows:", train_pd.shape[0])
    print("Train rows after outlier removal:", train_no_outliers_pd.shape[0])

    train_no_outliers = H2OFrame(train_no_outliers_pd)

    n_train_orig = train_pd.shape[0]
    n_train_clean = train_no_outliers_pd.shape[0]

    print("\n[INFO] Train-only target outlier removal")
    print(f"       Train rows before: {n_train_orig}")
    print(f"       Train rows after:  {n_train_clean} "
          f"({100.0 * n_train_clean / n_train_orig:.2f}% retained)")

    # Train model
    model = H2OGradientBoostingEstimator(**strong_params)

    start = time.time()
    model.train(
        x=x_cols,
        y=y_col,
        training_frame=train_no_outliers,
        validation_frame=None,  # no explicit valid; we’ll use test only
    )
    train_time = time.time() - start

    # For this experiment we just report test metrics (no CV)
    perf_test = model.model_performance(test_h2o)
    test_rmse = perf_test.rmse()
    test_mae = perf_test.mae()

    print(f"[{name}] Training time: {train_time:.2f} sec")
    print(f"[{name}] Test  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    # Get actual number of trees from the trained model
    try:
        n_trees_actual = int(model.params["ntrees"]["actual"])
    except Exception:
        n_trees_actual = strong_params.get("ntrees")

    result = {
        "experiment": name,
        "valid_RMSE": np.nan,
        "valid_MAE": np.nan,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "train_time_sec": train_time,
        "ntrees": strong_params.get("ntrees"),
        "max_depth": strong_params.get("max_depth"),
        "learn_rate": strong_params.get("learn_rate"),
        "sample_rate": strong_params.get("sample_rate"),
        "col_sample_rate": strong_params.get("col_sample_rate"),
        "min_rows": strong_params.get("min_rows"),
        "n_params": np.nan,
        "model": model,
    }
    return result


# ---------------------------------------------------------------------
# 5. Main: run everything & make plots
# ---------------------------------------------------------------------

def main():
    DATA_PATH = "../Part3/images/superconductivty-data/train.csv"

    # 1) Load & globally filter outliers (your original script)
    df = load_and_filter_outliers(
        data_path=DATA_PATH,
        target_col="critical_temp",
        iqr_k=1.5,
        apply_outlier_filter=True,
    )

    # 2) Shared train/test split (sklearn)
    X_train, X_test, y_train, y_test = train_test_split_shared(
        df,
        target_col="critical_temp",
        test_size=0.2,
        random_state=42,
    )

    # 3) Init H2O and create H2OFrames
    print("[INFO] Initializing H2O...")
    h2o.init()
    h2o.show_progress()

    train_h2o, test_h2o, x_cols, y_col = to_h2o_frames(
        X_train, X_test, y_train, y_test, target_col="critical_temp"
    )

    results: List[Dict[str, Any]] = []

    # Base GBM (with CV + early stopping)
    results.append(run_h2o_gbm_base(train_h2o, test_h2o, x_cols, y_col))

    # Manual GBM sweep: GBM_n800_d11 and GBM_VeryDeep_StrongRegular_ID5
    sweep_results = run_gbm_sweep(train_h2o, test_h2o, x_cols, y_col)
    results.extend(sweep_results)

    # "No Target Outliers" experiment (train-only filtering)
    strong_params = {
        "ntrees": 800,
        "max_depth": 12,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
        "seed": 42,
        "distribution": "gaussian",
    }
    results.append(
        run_gbm_no_target_outliers_train_only(
            "GBM_NoTargetOutliers_TrainOnly",
            strong_params,
            train_h2o,
            test_h2o,
            x_cols,
            y_col,
        )
    )

    # 4) Build summary table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_RMSE").reset_index(drop=True)

    cols_to_show = [
        "experiment",
        "valid_RMSE",
        "valid_MAE",
        "test_RMSE",
        "test_MAE",
        "train_time_sec",
        "n_trees",
        "max_depth",
        "learn_rate",
        "sample_rate",
        "col_sample_rate",
        "min_rows",
        "n_params",
    ]

    print("\n===== Final summary table (GBM variants) =====")
    print(results_df[cols_to_show])
    print("==============================================\n")

    results_df.to_csv("new_output/gbm_compare_repro_summary.csv", index=False)
    print("[INFO] Saved summary to gbm_compare_repro_summary.csv")

    # 5) Plots
    # RMSE plot
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["experiment"], results_df["test_RMSE"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Test RMSE")
    plt.title("GBM: Test RMSE comparison")
    plt.tight_layout()
    plt.savefig("new_output/gbm_compare_rmse.png", dpi=200)
    print("[INFO] Saved RMSE plot to gbm_compare_rmse.png")

    # Time plot
    plt.figure(figsize=(10, 5))
    plt.bar(results_df["experiment"], results_df["train_time_sec"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Training time (sec)")
    plt.title("GBM: Training time comparison")
    plt.tight_layout()
    plt.savefig("new_output/gbm_compare_time.png", dpi=200)
    print("[INFO] Saved training time plot to gbm_compare_time.png")

    # 6) Per-model diagnostic plots (pred vs actual, residuals)
    print("[INFO] Generating diagnostic plots for all models...")
    for res in results:
        exp_name = res["experiment"]
        model = res.get("model", None)
        if model is None:
            print(f"[WARN] No model stored for {exp_name}, skipping diagnostics.")
            continue
        plot_model_diagnostics(exp_name, model, test_h2o, y_col)


    # Optional: shutdown cluster
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()