#!/usr/bin/env python3
"""
Superconductivity GBM Experiments - Full Summary Script

This script:
- Loads train.csv (superconductivity dataset)
- Splits into train/valid/test
- Runs:
    * Baseline GBM
    * 12 manual GBM hyperparameter variants
    * GBM with standardized features
    * GBM with "Gaussianized" (log1p) features
    * PCA-based GBM
    * Top-K mutual information feature-selection GBM
    * GBM with outliers removed from the target
- Optionally runs a GBM grid search (RandomDiscrete) for tuning
- Records RMSE, MAE (validation & test), training time, and key params
- Prints a sorted summary table and top feature importances for the best model
- Optionally generates multiple evaluation plots for all models
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import h2o
from h2o import H2OFrame
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch


# -------------------------------------------------------------------
# Config flags
# -------------------------------------------------------------------

RUN_GRID_SEARCH = False  # set to True if you want to re-run the tuned GBM grid search
PLOT_RESULTS = True      # set to False if you don't want any plots
PLOT_SPLIT = "valid"     # which split to use for plots: "train", "valid", or "test"

OUTPUT_DIR = "Part3/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------

def compute_metrics(results_dict, split="valid"):
    """Return dict of RMSE/MAE/R2 for each model on a given split."""
    metrics = {}
    for model_name, splits in results_dict.items():
        y_true = np.array(splits[split]["y_true"])
        y_pred = np.array(splits[split]["y_pred"])

        # Older sklearn versions don't support `squared=False`,
        # so we compute RMSE manually from MSE.
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics[model_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    return metrics

def plot_rmse_mae_bar(results_dict, split="valid"):
    metrics = compute_metrics(results_dict, split=split)
    model_names = list(metrics.keys())
    rmse_vals = [metrics[m]["RMSE"] for m in model_names]
    mae_vals = [metrics[m]["MAE"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, rmse_vals, width, label="RMSE")
    plt.bar(x + width / 2, mae_vals, width, label="MAE")

    plt.xticks(x, model_names, rotation=20)
    plt.ylabel("Error")
    plt.title(f"RMSE and MAE by Model ({split} set)")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/rmse_mae_bar.png', dpi=300)

def plot_pred_vs_actual(results_dict, split="valid"):
    plt.figure(figsize=(8, 8))

    for model_name, splits in results_dict.items():
        y_true = np.array(splits[split]["y_true"])
        y_pred = np.array(splits[split]["y_pred"])
        plt.scatter(y_true, y_pred, alpha=0.4, label=model_name)

    # Diagonal line (perfect prediction)
    all_true = np.concatenate(
        [np.array(results_dict[m][split]["y_true"]) for m in results_dict]
    )
    min_val, max_val = all_true.min(), all_true.max()
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("Actual critical_temp")
    plt.ylabel("Predicted critical_temp")
    plt.title(f"Predicted vs Actual ({split} set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/prediction_vs_actual.png', dpi=300)

def plot_residual_scatter(results_dict, split="valid"):
    plt.figure(figsize=(8, 6))

    for model_name, splits in results_dict.items():
        y_true = np.array(splits[split]["y_true"])
        y_pred = np.array(splits[split]["y_pred"])
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.4, label=model_name)

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted critical_temp")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted ({split} set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/residual_scatter.png', dpi=300)


def plot_residual_hist(results_dict, split="valid", bins=50):
    plt.figure(figsize=(8, 6))

    for model_name, splits in results_dict.items():
        y_true = np.array(splits[split]["y_true"])
        y_pred = np.array(splits[split]["y_pred"])
        residuals = y_true - y_pred
        plt.hist(residuals, bins=bins, alpha=0.4, label=model_name)

    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution ({split} set)")
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/residual_history.png', dpi=300)

def plot_error_vs_target(results_dict, split="valid"):
    plt.figure(figsize=(8, 6))

    for model_name, splits in results_dict.items():
        y_true = np.array(splits[split]["y_true"])
        y_pred = np.array(splits[split]["y_pred"])
        abs_err = np.abs(y_true - y_pred)
        plt.scatter(y_true, abs_err, alpha=0.4, label=model_name)

    plt.xlabel("Actual critical_temp")
    plt.ylabel("|Error| = |y_true - y_pred|")
    plt.title(f"Absolute Error vs critical_temp ({split} set)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/error_vs_target.png', dpi=300)

def plot_h2o_learning_curve(h2o_models, split_name="validation"):
    """
    h2o_models: dict like {"GBM_base": gbm_base, "GBM_PCA": gbm_pca, ...}
    split_name: 'training' or 'validation' depending on column names in scoring_history
    """
    plt.figure(figsize=(10, 6))

    for model_name, model in h2o_models.items():
        sh = model.scoring_history().as_data_frame()
        # Common GBM columns: 'number_of_trees', 'training_rmse', 'validation_rmse'
        if split_name == "training":
            col = "training_rmse"
        else:
            col = "validation_rmse"

        if col not in sh.columns:
            print(f"[WARN] {col} not found for {model_name}; available: {sh.columns.tolist()}")
            continue

        x_col = "number_of_trees" if "number_of_trees" in sh.columns else sh.columns[0]
        plt.plot(sh[x_col], sh[col], label=f"{model_name} ({split_name})")

    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE")
    plt.title(f"H2O GBM Learning Curves ({split_name} RMSE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/h2o_learning_curve.png', dpi=300)

def plot_h2o_feature_importance(model, top_n=20, title_suffix=""):
    """Horizontal bar plot of H2O GBM feature importances."""
    varimp = model.varimp(use_pandas=True)
    varimp_top = varimp.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(varimp_top["variable"], varimp_top["relative_importance"])
    plt.gca().invert_yaxis()  # most important at top
    plt.xlabel("Relative Importance")
    plt.title(f"Top {top_n} Features {title_suffix}")
    plt.tight_layout()
    #plt.show()
    plt.savefig('output/h2o_feature_importance.png', dpi=300)

# -------------------------------------------------------------------
# Helper: run a GBM experiment with timing & metrics
# -------------------------------------------------------------------

def run_gbm_experiment(
    name,
    train_frame,
    valid_frame,
    test_frame,
    feature_cols,
    target_col,
    gbm_kwargs
):
    """
    Train a GBM with the given kwargs, measure training time,
    and compute RMSE/MAE on valid & test sets.

    Returns:
        result dict including metrics, time, model, and the frames used.
    """
    print(f"\n===== Running experiment: {name} =====")
    model = H2OGradientBoostingEstimator(**gbm_kwargs)

    start = time.time()
    model.train(
        x=feature_cols,
        y=target_col,
        training_frame=train_frame,
        validation_frame=valid_frame
    )
    end = time.time()
    train_time = end - start

    perf_valid = model.model_performance(valid_frame)
    perf_test = model.model_performance(test_frame)

    valid_rmse = perf_valid.rmse()
    valid_mae = perf_valid.mae()
    test_rmse = perf_test.rmse()
    test_mae = perf_test.mae()

    print(f"[{name}] Training time: {train_time:.2f} sec")
    print(f"[{name}] Valid RMSE={valid_rmse:.4f}, MAE={valid_mae:.4f}")
    print(f"[{name}] Test  RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")

    result = {
        "experiment": name,
        "valid_RMSE": valid_rmse,
        "valid_MAE": valid_mae,
        "test_RMSE": test_rmse,
        "test_MAE": test_mae,
        "train_time_sec": train_time,
        "ntrees": gbm_kwargs.get("ntrees"),
        "max_depth": gbm_kwargs.get("max_depth"),
        "learn_rate": gbm_kwargs.get("learn_rate"),
        "sample_rate": gbm_kwargs.get("sample_rate"),
        "col_sample_rate": gbm_kwargs.get("col_sample_rate"),
        "min_rows": gbm_kwargs.get("min_rows"),
        "model": model,
        "valid_frame": valid_frame,
        "test_frame": test_frame,
        "target_col": target_col,
    }
    return result


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    h2o.init()

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------
    print("Importing train.csv...")
    data = h2o.import_file("../Part3/images/superconductivty-data/train.csv")

    y = "critical_temp"
    X = data.col_names
    X.remove(y)

    # Split into train / valid / test
    train, valid, test = data.split_frame(ratios=[0.7, 0.15], seed=176)
    print("Train rows:", train.nrows)
    print("Valid rows:", valid.nrows)
    print("Test rows:", test.nrows)

    # Also keep pandas versions for transformations
    train_pd = train.as_data_frame()
    valid_pd = valid.as_data_frame()
    test_pd = test.as_data_frame()

    results = []

    # ---------------------------------------------------------------
    # 1. Baseline GBM (ID 1)
    # ---------------------------------------------------------------
    baseline_params = dict(
        ntrees=100,
        max_depth=5,
        learn_rate=0.10,
        sample_rate=1.0,
        col_sample_rate=1.0,
        min_rows=10,
        stopping_rounds=5,
        stopping_metric="RMSE",
        stopping_tolerance=1e-4,
        seed=176,
    )
    results.append(run_gbm_experiment(
        "GBM Baseline (ID 1)",
        train, valid, test, X, y, baseline_params
    ))

    # ---------------------------------------------------------------
    # 2. Optional: GBM Tuned (Grid Search)
    # ---------------------------------------------------------------
    if RUN_GRID_SEARCH:
        print("\n===== Running GBM Grid Search (Tuned) =====")
        hyper_params = {
            "ntrees":            [100, 300, 500, 800],
            "max_depth":         [4, 6, 8, 10, 12],
            "learn_rate":        [0.01, 0.03, 0.05, 0.1],
            "sample_rate":       [0.7, 0.8, 0.9, 1.0],
            "col_sample_rate":   [0.7, 0.8, 0.9, 1.0],
            "min_rows":          [1, 5, 10, 20]
        }

        search_criteria = {
            "strategy": "RandomDiscrete",
            "max_models": 30,
            "seed": 176,
            "stopping_rounds": 5,
            "stopping_metric": "RMSE",
            "stopping_tolerance": 1e-4
        }

        gbm_grid = H2OGridSearch(
            model=H2OGradientBoostingEstimator(
                nfolds=0,
                seed=176,
                stopping_rounds=5,
                stopping_metric="RMSE",
                stopping_tolerance=1e-4
            ),
            hyper_params=hyper_params,
            search_criteria=search_criteria
        )

        start = time.time()
        gbm_grid.train(x=X, y=y, training_frame=train, validation_frame=valid)
        end = time.time()
        grid_time = end - start

        sorted_grid = gbm_grid.get_grid(sort_by="rmse", decreasing=False)
        best_model = sorted_grid.models[0]
        print(sorted_grid)

        perf_valid = best_model.model_performance(valid)
        perf_test = best_model.model_performance(test)

        results.append({
            "experiment": "GBM Tuned (Grid Search)",
            "valid_RMSE": perf_valid.rmse(),
            "valid_MAE": perf_valid.mae(),
            "test_RMSE": perf_test.rmse(),
            "test_MAE": perf_test.mae(),
            "train_time_sec": grid_time,
            "ntrees": best_model.get_params().get("ntrees", None),
            "max_depth": best_model.get_params().get("max_depth", None),
            "learn_rate": best_model.get_params().get("learn_rate", None),
            "sample_rate": best_model.get_params().get("sample_rate", None),
            "col_sample_rate": best_model.get_params().get("col_sample_rate", None),
            "min_rows": best_model.get_params().get("min_rows", None),
            "model": best_model,
            "valid_frame": valid,
            "test_frame": test,
            "target_col": y,
        })

    # ---------------------------------------------------------------
    # 3. 12 Manual GBM Variants (IDs 1–12 from your table)
    # ---------------------------------------------------------------
    manual_configs = [
        # ID,  name, ntrees, max_depth, learn_rate, sample_rate, col_sample_rate, min_rows
        (2,  "Shallow, many trees (ID 2)",          500, 4,  0.05, 1.0, 1.0, 10),
        # (3,  "Deep, fewer trees (ID 3)",            200, 10, 0.05, 1.0, 1.0, 5),
        # (4,  "Deep, many trees (ID 4)",             800, 10, 0.03, 1.0, 1.0, 5),
        # (5,  "Very deep, strong regular (ID 5)",    800, 12, 0.03, 0.8, 0.8, 10),
        # (6,  "Small LR, many trees (ID 6)",        1000, 8,  0.01, 0.9, 0.9, 5),
        # (7,  "Higher LR, fewer trees (ID 7)",       200, 8,  0.10, 0.8, 0.8, 10),
        # (8,  "Row subsampling focus (ID 8)",        500, 8,  0.05, 0.7, 1.0, 5),
        # (9,  "Column subsampling focus (ID 9)",     500, 8,  0.05, 1.0, 0.7, 5),
        # (10, "Strong subsampling both (ID 10)",     800, 8,  0.05, 0.7, 0.7, 5),
        # (11, "Regularized via min_rows (ID 11)",    500, 8,  0.05, 0.9, 0.9, 20),
        # (12, "Compact model (fast) (ID 12)",        150, 6,  0.07, 0.9, 0.9, 10),
    ]

    for (id_num, name, ntrees, max_depth, lr, samp, col_samp, min_rows) in manual_configs:
        params = dict(
            ntrees=ntrees,
            max_depth=max_depth,
            learn_rate=lr,
            sample_rate=samp,
            col_sample_rate=col_samp,
            min_rows=min_rows,
            stopping_rounds=5,
            stopping_metric="RMSE",
            stopping_tolerance=1e-4,
            seed=176,
        )
        results.append(run_gbm_experiment(
            f"GBM Manual {id_num}: {name}",
            train, valid, test, X, y, params
        ))

    # ---------------------------------------------------------------
    # 4. GBM with Standardized Features
    # ---------------------------------------------------------------
    print("\n===== Building standardized (z-score) features =====")
    X_cols = X  # just to make naming consistent

    scaler_std = StandardScaler()
    X_train_std = scaler_std.fit_transform(train_pd[X_cols])
    X_valid_std = scaler_std.transform(valid_pd[X_cols])
    X_test_std = scaler_std.transform(test_pd[X_cols])

    train_std_pd = pd.DataFrame(X_train_std, columns=X_cols)
    valid_std_pd = pd.DataFrame(X_valid_std, columns=X_cols)
    test_std_pd = pd.DataFrame(X_test_std, columns=X_cols)

    train_std_pd[y] = train_pd[y].values
    valid_std_pd[y] = valid_pd[y].values
    test_std_pd[y] = test_pd[y].values

    train_std = H2OFrame(train_std_pd)
    valid_std = H2OFrame(valid_std_pd)
    test_std = H2OFrame(test_std_pd)

    strong_params = dict(
        ntrees=800,
        max_depth=12,
        learn_rate=0.03,
        sample_rate=0.8,
        col_sample_rate=0.8,
        min_rows=10,
        stopping_rounds=5,
        stopping_metric="RMSE",
        stopping_tolerance=1e-4,
        seed=176,
    )
    results.append(run_gbm_experiment(
        "GBM (Standardized Features)",
        train_std, valid_std, test_std, X_cols, y, strong_params
    ))

    # ---------------------------------------------------------------
    # 5. GBM with "Gaussianized" Features (log1p)
    # ---------------------------------------------------------------
    print("\n===== Building Gaussianized (log1p) features =====")
    train_gauss_pd = train_pd.copy()
    valid_gauss_pd = valid_pd.copy()
    test_gauss_pd = test_pd.copy()

    shifts = {}
    for col in X_cols:
        min_val = train_gauss_pd[col].min()
        shift = 1.0 - min_val if min_val <= 0 else 0.0
        shifts[col] = shift
        train_gauss_pd[col] = np.log1p(train_gauss_pd[col] + shift)
        valid_gauss_pd[col] = np.log1p(valid_gauss_pd[col] + shift)
        test_gauss_pd[col] = np.log1p(test_gauss_pd[col] + shift)

    train_gauss = H2OFrame(train_gauss_pd)
    valid_gauss = H2OFrame(valid_gauss_pd)
    test_gauss = H2OFrame(test_gauss_pd)

    results.append(run_gbm_experiment(
        "GBM (Gaussianized Features)",
        train_gauss, valid_gauss, test_gauss, X_cols, y, strong_params
    ))

    # ---------------------------------------------------------------
    # 6. PCA-based GBM (on standardized features)
    # ---------------------------------------------------------------
    print("\n===== PCA-based GBM =====")
    scaler_pca = StandardScaler()
    X_train_scaled = scaler_pca.fit_transform(train_pd[X_cols])
    X_valid_scaled = scaler_pca.transform(valid_pd[X_cols])
    X_test_scaled = scaler_pca.transform(test_pd[X_cols])

    pca = PCA(n_components=0.95, random_state=176)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_valid_pca = pca.transform(X_valid_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print("Original dim:", X_train_scaled.shape[1])
    print("PCA dim:", X_train_pca.shape[1])

    pc_names = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

    train_pca_pd = pd.DataFrame(X_train_pca, columns=pc_names)
    valid_pca_pd = pd.DataFrame(X_valid_pca, columns=pc_names)
    test_pca_pd = pd.DataFrame(X_test_pca, columns=pc_names)

    train_pca_pd[y] = train_pd[y].values
    valid_pca_pd[y] = valid_pd[y].values
    test_pca_pd[y] = test_pd[y].values

    train_pca = H2OFrame(train_pca_pd)
    valid_pca = H2OFrame(valid_pca_pd)
    test_pca = H2OFrame(test_pca_pd)

    results.append(run_gbm_experiment(
        "GBM (PCA 95% variance)",
        train_pca, valid_pca, test_pca, pc_names, y, strong_params
    ))

    def summarize_pca(pca_obj, feature_names, n_top=5):
        """
        Summarize a fitted PCA object:
          - prints original vs PCA dims
          - prints explained variance per component + cumulative
          - prints top-contributing original features per PC
        """
        n_features = len(feature_names)
        n_components = pca_obj.n_components_

        print("\n===== PCA SUMMARY =====")
        print(f"Original features:  {n_features}")
        print(f"PCA components:     {n_components}\n")

        # 1) Variance explained per component
        var_ratio = pca_obj.explained_variance_ratio_
        cum_var = np.cumsum(var_ratio)

        var_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(n_components)],
            "Explained_Variance_Ratio": var_ratio,
            "Cumulative_Variance": cum_var
        })
        print("=== Variance Explained by Each PC ===")
        print(var_df)
        print("\nTotal variance captured by PCA:",
              round(cum_var[-1], 4))

        # 2) Loadings (how much each feature contributes to each PC)
        loadings = pd.DataFrame(
            pca_obj.components_.T,              # shape: (n_features, n_components)
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(n_components)]
        )

        # 3) Top contributing features per PC
        print("\n=== Top contributing features for each PC ===")
        for pc in loadings.columns:
            print(f"\n{pc}:")
            top_pos = loadings[pc].sort_values(ascending=False).head(n_top)
            top_neg = loadings[pc].sort_values(ascending=True).head(n_top)

            print("  Top positive contributors:")
            for feat, val in top_pos.items():
                print(f"    {feat:35s} {val: .4f}")

            print("  Top negative contributors:")
            for feat, val in top_neg.items():
                print(f"    {feat:35s} {val: .4f}")

    summarize_pca(pca, X_cols)

    # ---------------------------------------------------------------
    # 7. Top-K Mutual Information Feature Selection + GBM
    # ---------------------------------------------------------------
    print("\n===== Top-K (MI) feature selection + GBM =====")
    mi = mutual_info_regression(train_pd[X_cols], train_pd[y])
    mi_series = pd.Series(mi, index=X_cols).sort_values(ascending=False)

    print("Top 20 features by mutual information:")
    print(mi_series.head(20))

    K = 20
    topK = mi_series.head(K).index.tolist()

    train_top_pd = train_pd[topK + [y]]
    valid_top_pd = valid_pd[topK + [y]]
    test_top_pd = test_pd[topK + [y]]

    train_top = H2OFrame(train_top_pd)
    valid_top = H2OFrame(valid_top_pd)
    test_top = H2OFrame(test_top_pd)

    results.append(run_gbm_experiment(
        f"GBM (Top-{K} MI Features)",
        train_top, valid_top, test_top, topK, y, strong_params
    ))

    # ---------------------------------------------------------------
    # 8. GBM with Target Outliers Removed (IQR on train)
    # ---------------------------------------------------------------
    print("\n===== GBM with target outliers removed (IQR filter) =====")
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

    results.append(run_gbm_experiment(
        "GBM (No Target Outliers)",
        train_no_outliers, valid, test, X_cols, y, strong_params
    ))

    # ---------------------------------------------------------------
    # 9. Build final summary table
    # ---------------------------------------------------------------
    print("\n===== Building final summary table =====")
    # Drop model and frames for DataFrame
    results_for_df = []
    for r in results:
        r_copy = {k: v for k, v in r.items()
                  if k not in ("model", "valid_frame", "test_frame", "target_col")}
        results_for_df.append(r_copy)

    results_df = pd.DataFrame(results_for_df)
    results_df_sorted = results_df.sort_values(by="test_RMSE").reset_index(drop=True)
    print(results_df_sorted)

    summary_path = os.path.join(OUTPUT_DIR, "gbm_experiments_summary.csv")
    results_df_sorted.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # ---------------------------------------------------------------
    # 10. Feature importance for best model
    # ---------------------------------------------------------------
    best_exp_name = results_df_sorted.iloc[0]["experiment"]
    print(f"\n===== Feature Importance for best model: {best_exp_name} =====")

    # Find the corresponding model object
    best_model_obj = None
    for r in results:
        if r["experiment"] == best_exp_name:
            best_model_obj = r["model"]
            break

    if best_model_obj is not None:
        try:
            varimp_df = best_model_obj.varimp(use_pandas=True)
            print("\nTop 20 features/components by importance:")
            print(varimp_df.head(20))
        except Exception as e:
            print("Could not compute variable importance:", e)
    else:
        print("Best model object not found (this should not happen).")

    # ---------------------------------------------------------------
    # 11. Build structures for plotting & generate plots
    # ---------------------------------------------------------------
    if PLOT_RESULTS:
        print("\n===== Building plot_results & generating plots =====")

        plot_results = {}
        h2o_models = {}

        for r in results:
            name = r["experiment"]
            model = r["model"]
            valid_frame = r["valid_frame"]
            test_frame = r["test_frame"]
            target_col = r["target_col"]

            # Get pandas versions for y_true
            valid_pd_local = valid_frame.as_data_frame()
            test_pd_local = test_frame.as_data_frame()

            y_valid_true = valid_pd_local[target_col].values
            y_test_true = test_pd_local[target_col].values

            # Predictions
            valid_pred = model.predict(valid_frame).as_data_frame()["predict"].values
            test_pred = model.predict(test_frame).as_data_frame()["predict"].values

            plot_results[name] = {
                "valid": {"y_true": y_valid_true, "y_pred": valid_pred},
                "test": {"y_true": y_test_true, "y_pred": test_pred},
                # If you want train later, you could add it too
            }

            h2o_models[name] = model

        split = PLOT_SPLIT  # "valid" by default
        print(f"Creating plots using split = '{split}'")

        # High-level comparison
        plot_rmse_mae_bar(plot_results, split=split)

        # Fit quality plots
        plot_pred_vs_actual(plot_results, split=split)
        plot_residual_scatter(plot_results, split=split)
        plot_residual_hist(plot_results, split=split)
        plot_error_vs_target(plot_results, split=split)

        # H2O-specific: learning curves
        plot_h2o_learning_curve(h2o_models, split_name="training")
        plot_h2o_learning_curve(h2o_models, split_name="validation")

        # Feature importance bar plot for best model
        if best_model_obj is not None:
            plot_h2o_feature_importance(
                best_model_obj, top_n=20, title_suffix=f"({best_exp_name})"
            )


if __name__ == "__main__":
    main()