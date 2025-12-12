#!/usr/bin/env python3
"""
Unified Experiments for Superconductivity Dataset (Extended)

Models:
  - H2O GBM (base config)
  - H2O Random Forest
  - H2O GLM (linear regression with regularization)
  - Scikit-learn SVR (RBF) with PCA

Extras:
  - Optional GBM hyperparameter sweep (manual configs from earlier experiments)
  - 5-fold cross-validation for all models
  - RMSE & MAE on CV and test
  - Approximate parameter / complexity counts
  - Automatic anomaly tagging (notes column)
  - Results saved to CSV and printed sorted by Test RMSE
"""

import argparse
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR

import h2o
from h2o.estimators import (
    H2OGradientBoostingEstimator,
    H2ORandomForestEstimator,
    H2OGeneralizedLinearEstimator,
)

# ------------------------------------------------------------------------
# GBM sweep configurations (based on your earlier manual experiments)
# ------------------------------------------------------------------------

GBM_SWEEP_CONFIGS = [
    # Original Baseline (ID 1)
    {
        "name": "GBM_Baseline_ID1",
        "ntrees": 100,
        "max_depth": 5,
        "learn_rate": 0.10,
        "sample_rate": 1.0,
        "col_sample_rate": 1.0,
        "min_rows": 10,
    },
    # Shallow, many trees (ID 2)
    {
        "name": "GBM_Shallow_ManyTrees_ID2",
        "ntrees": 500,
        "max_depth": 4,
        "learn_rate": 0.05,
        "sample_rate": 1.0,
        "col_sample_rate": 1.0,
        "min_rows": 10,
    },
    # Deep, fewer trees (ID 3)
    {
        "name": "GBM_Deep_FewerTrees_ID3",
        "ntrees": 200,
        "max_depth": 10,
        "learn_rate": 0.05,
        "sample_rate": 1.0,
        "col_sample_rate": 1.0,
        "min_rows": 5,
    },
    # Deep, many trees (ID 4)
    {
        "name": "GBM_Deep_ManyTrees_ID4",
        "ntrees": 800,
        "max_depth": 10,
        "learn_rate": 0.03,
        "sample_rate": 1.0,
        "col_sample_rate": 1.0,
        "min_rows": 5,
    },
    # Very deep, strong regular (ID 5)
    {
        "name": "GBM_VeryDeep_StrongRegular_ID5",
        "ntrees": 800,
        "max_depth": 12,
        "learn_rate": 0.03,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    # Small LR, many trees (ID 6)
    {
        "name": "GBM_SmallLR_ManyTrees_ID6",
        "ntrees": 1000,
        "max_depth": 8,
        "learn_rate": 0.01,
        "sample_rate": 0.9,
        "col_sample_rate": 0.9,
        "min_rows": 5,
    },
    # Higher LR, fewer trees (ID 7)
    {
        "name": "GBM_HighLR_FewerTrees_ID7",
        "ntrees": 200,
        "max_depth": 8,
        "learn_rate": 0.10,
        "sample_rate": 0.8,
        "col_sample_rate": 0.8,
        "min_rows": 10,
    },
    # Row subsampling focus (ID 8)
    {
        "name": "GBM_RowSubsampling_ID8",
        "ntrees": 500,
        "max_depth": 8,
        "learn_rate": 0.05,
        "sample_rate": 0.7,
        "col_sample_rate": 1.0,
        "min_rows": 5,
    },
    # Column subsampling focus (ID 9)
    {
        "name": "GBM_ColSubsampling_ID9",
        "ntrees": 500,
        "max_depth": 8,
        "learn_rate": 0.05,
        "sample_rate": 1.0,
        "col_sample_rate": 0.7,
        "min_rows": 5,
    },
    # Strong subsampling both (ID 10)
    {
        "name": "GBM_StrongSubsampling_ID10",
        "ntrees": 800,
        "max_depth": 8,
        "learn_rate": 0.05,
        "sample_rate": 0.7,
        "col_sample_rate": 0.7,
        "min_rows": 5,
    },
    # Regularized via min_rows (ID 11)
    {
        "name": "GBM_Reg_MinRows_ID11",
        "ntrees": 500,
        "max_depth": 8,
        "learn_rate": 0.05,
        "sample_rate": 0.9,
        "col_sample_rate": 0.9,
        "min_rows": 20,
    },
    # Compact model (fast) (ID 12)
    {
        "name": "GBM_Compact_Fast_ID12",
        "ntrees": 150,
        "max_depth": 6,
        "learn_rate": 0.07,
        "sample_rate": 0.9,
        "col_sample_rate": 0.9,
        "min_rows": 10,
    },
]

# ------------------------------------------------------------------------
# Data loading & preprocessing
# ------------------------------------------------------------------------

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
        X_train, X_test, y_train, y_test (numpy/pandas)
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


# ------------------------------------------------------------------------
# H2O helpers
# ------------------------------------------------------------------------

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
    Approximate complexity for tree-based H2O models (GBM / RF).
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


def h2o_glm_complexity(model) -> Dict[str, Any]:
    """
    Count GLM parameters as number of coefficients (including intercept).
    """
    try:
        coefs = model.coef()
        n_params = len(coefs)
        return {
            "n_params": n_params,
        }
    except Exception as e:
        print(f"[WARN] Could not compute GLM complexity: {e}")
        return {
            "n_params": np.nan,
        }


# ------------------------------------------------------------------------
# Model training functions (base models)
# ------------------------------------------------------------------------

def run_h2o_gbm(train, test, x_cols, y_col, seed=42) -> Dict[str, Any]:
    """
    Train base GBM with 5-fold CV and return metrics + complexity.
    """
    print("[INFO] Training H2O GBM (base config)...")
    start = time.perf_counter()

    gbm = H2OGradientBoostingEstimator(
        ntrees=500,
        max_depth=8,
        learn_rate=0.05,
        sample_rate=0.8,
        col_sample_rate=0.8,
        min_rows=5,
        nfolds=5,
        seed=seed,
        keep_cross_validation_models=True,
    )
    gbm.train(x=x_cols, y=y_col, training_frame=train)

    elapsed = time.perf_counter() - start
    metrics = get_h2o_cv_and_test_metrics(gbm, test)
    complexity = h2o_tree_model_complexity(gbm)

    result = {
        "model_name": "H2O_GBM",
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
    }
    return result


def run_h2o_rf(train, test, x_cols, y_col, seed=42) -> Dict[str, Any]:
    """
    Train Random Forest with 5-fold CV and return metrics + complexity.
    """
    print("[INFO] Training H2O Random Forest...")
    start = time.perf_counter()

    rf = H2ORandomForestEstimator(
        ntrees=500,
        max_depth=20,
        sample_rate=0.8,
        col_sample_rate_per_tree=0.8,
        min_rows=5,
        nfolds=5,
        seed=seed,
        keep_cross_validation_models=True,
    )
    rf.train(x=x_cols, y=y_col, training_frame=train)

    elapsed = time.perf_counter() - start
    metrics = get_h2o_cv_and_test_metrics(rf, test)
    complexity = h2o_tree_model_complexity(rf)

    result = {
        "model_name": "H2O_RF",
        "algorithm": "RandomForest",
        "train_time_sec": elapsed,
        "cv_rmse": metrics["cv_rmse"],
        "cv_mae": metrics["cv_mae"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "n_params": complexity["approx_params"],
        "n_trees": complexity["n_trees"],
        "max_depth": complexity["max_depth"],
        "mean_depth": complexity["mean_depth"],
    }
    return result


def run_h2o_glm(train, test, x_cols, y_col, seed=42) -> Dict[str, Any]:
    """
    Train GLM (linear regression with regularization) with 5-fold CV.
    """
    print("[INFO] Training H2O GLM (linear)...")
    start = time.perf_counter()

    glm = H2OGeneralizedLinearEstimator(
        family="gaussian",
        lambda_search=True,
        alpha=[0.0, 0.5, 1.0],
        nfolds=5,
        seed=seed,
        standardize=True,
        solver="IRLSM",
    )
    glm.train(x=x_cols, y=y_col, training_frame=train)

    elapsed = time.perf_counter() - start
    metrics = get_h2o_cv_and_test_metrics(glm, test)
    complexity = h2o_glm_complexity(glm)

    result = {
        "model_name": "H2O_GLM",
        "algorithm": "LinearRegression(GLM)",
        "train_time_sec": elapsed,
        "cv_rmse": metrics["cv_rmse"],
        "cv_mae": metrics["cv_mae"],
        "test_rmse": metrics["test_rmse"],
        "test_mae": metrics["test_mae"],
        "n_params": complexity["n_params"],
        "n_trees": np.nan,
        "max_depth": np.nan,
        "mean_depth": np.nan,
    }
    return result


def run_svr_with_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    n_components: int = 50,
    cv_folds: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train an SVR (RBF kernel) with a PCA+StandardScaler pipeline.
    Uses sklearn K-fold CV on the train set.

    Complexity is approximated as:
        n_params ≈ #support_vectors * (n_components + 1)
    """
    print("[INFO] Training SVR (RBF) with PCA...")
    start = time.perf_counter()

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=seed)),
            ("svr", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)),
        ]
    )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    cv_rmse_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    cv_mae_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    pipeline.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    y_pred_test = pipeline.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mse_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    svr_model = pipeline.named_steps["svr"]
    pca_model = pipeline.named_steps["pca"]

    try:
        n_support = svr_model.support_vectors_.shape[0]
        n_features_pca = pca_model.n_components_
        approx_params = n_support * (n_features_pca + 1)
    except Exception as e:
        print(f"[WARN] Could not extract SVR complexity: {e}")
        n_support = np.nan
        approx_params = np.nan

    result = {
        "model_name": "SVR_PCA",
        "algorithm": "SVR(RBF)+PCA",
        "train_time_sec": elapsed,
        "cv_rmse": -np.mean(cv_rmse_scores),
        "cv_mae": -np.mean(cv_mae_scores),
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "n_params": approx_params,
        "n_support_vectors": n_support,
        "n_trees": np.nan,
        "max_depth": np.nan,
        "mean_depth": np.nan,
    }
    return result


# ------------------------------------------------------------------------
# GBM hyperparameter sweep
# ------------------------------------------------------------------------

def run_gbm_sweep(train, test, x_cols, y_col, seed=42) -> List[Dict[str, Any]]:
    """
    Run a sweep of manual GBM configurations (from earlier experiments).
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
            keep_cross_validation_models=False,
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
        })

    return sweep_results


# ------------------------------------------------------------------------
# Anomaly tagging
# ------------------------------------------------------------------------

def annotate_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'notes' column flagging interesting / anomalous behaviors:
      - very_high_error: test_rmse much worse than best model
      - very_large_model: n_params > 500k
      - very_slow_training: train_time_sec > 120s
      - cv_vs_test_mismatch: |cv_rmse - test_rmse| > 1.0
    """
    df = df.copy()
    best_rmse = df["test_rmse"].min()
    notes = []

    for _, row in df.iterrows():
        flags = []

        if row["test_rmse"] > 1.5 * best_rmse:
            flags.append("very_high_error")

        n_params = row.get("n_params", np.nan)
        if not pd.isna(n_params) and n_params > 500_000:
            flags.append("very_large_model")

        if row["train_time_sec"] > 120:
            flags.append("very_slow_training")

        cv_rmse = row.get("cv_rmse", np.nan)
        test_rmse = row.get("test_rmse", np.nan)
        if not pd.isna(cv_rmse) and not pd.isna(test_rmse):
            if abs(cv_rmse - test_rmse) > 1.0:
                flags.append("cv_vs_test_mismatch")

        notes.append(",".join(flags) if flags else "")

    df["notes"] = notes
    return df


# ------------------------------------------------------------------------
# Main experiment runner
# ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified Superconductivity Regression Experiments (Extended)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="images/superconductivty-data/train.csv",
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
        default="output/results_superconductivity_models.csv",
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

    # 2. Shared train/test split (for all models)
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

    # 4. Base models
    results.append(run_h2o_gbm(train_h2o, test_h2o, x_cols, y_col))
    results.append(run_h2o_rf(train_h2o, test_h2o, x_cols, y_col))
    results.append(run_h2o_glm(train_h2o, test_h2o, x_cols, y_col))
    results.append(run_svr_with_pca(X_train, X_test, y_train, y_test))

    # 5. Optional GBM hyperparameter sweep
    if args.run_gbm_sweep:
        sweep_results = run_gbm_sweep(train_h2o, test_h2o, x_cols, y_col)
        results.extend(sweep_results)

    # 6. Results table + anomalies
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_rmse").reset_index(drop=True)
    results_df = annotate_anomalies(results_df)

    print("\n====================== MODEL COMPARISON (Sorted by Test RMSE) ======================")
    cols_to_show = [
        "model_name",
        "algorithm",
        "cv_rmse",
        "test_rmse",
        "cv_mae",
        "test_mae",
        "n_params",
        "n_trees",
        "train_time_sec",
        "notes",
    ]
    print(results_df[cols_to_show])
    print("====================================================================================\n")

    results_df.to_csv(args.output_csv, index=False)
    print(f"[INFO] Saved results to {args.output_csv}")

    # Shutdown H2O (optional)
    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()