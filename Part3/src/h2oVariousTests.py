#!/usr/bin/env python3
"""
Stroke Dataset GBM Experiments

- Loads stroke dataset
- Splits into train/validation/test
- Runs:
    * Baseline GBM
    * Manual GBM hyperparameter variants
    * GBM with standardized features
    * GBM with "Gaussianized" features (log1p)
    * PCA-based GBM (numeric columns)
- Records AUC, logloss, F1, training time, and GBM params
- Prints sorted summary table and feature importance for best model
"""

import os
import time
import numpy as np
import pandas as pd

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------
# Config
# -------------------------------
OUTPUT_DIR = "../output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Helper: run GBM classification
# -------------------------------
def run_gbm_classification(name, train_frame, valid_frame, test_frame, feature_cols, target_col, gbm_kwargs):
    print(f"\n===== Running experiment: {name} =====")
    start = time.time()
    model = H2OGradientBoostingEstimator(**gbm_kwargs)
    model.train(x=feature_cols, y=target_col, training_frame=train_frame, validation_frame=valid_frame)
    end = time.time()

    perf_valid = model.model_performance(valid_frame)
    perf_test  = model.model_performance(test_frame)

    result = {
        "experiment": name,
        "valid_AUC": perf_valid.auc(),
        "valid_logloss": perf_valid.logloss(),
        "valid_F1": perf_valid.F1()[0][1],
        "test_AUC": perf_test.auc(),
        "test_logloss": perf_test.logloss(),
        "test_F1": perf_test.F1()[0][1],
        "train_time_sec": end - start,
        "model": model,
        **gbm_kwargs
    }
    return result

# -------------------------------
# Main
# -------------------------------
def main():
    h2o.init(max_mem_size="4G")

    # Load dataset
    data_path = "data/healthcare-dataset-stroke-data.csv"
    data = h2o.import_file(data_path)

    # Drop rows with missing target
    data = data[~data["stroke"].isna()]

    # Categorical columns
    categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
    for col in categorical_cols:
        data[col] = data[col].asfactor()

    # Target
    y = "stroke"
    data[y] = data[y].asfactor()

    # Features
    X = [c for c in data.columns if c != y]

    # Split
    train, valid, test = data.split_frame(ratios=[0.7, 0.15], seed=176)

    # Keep pandas versions for transformations
    train_pd = train.as_data_frame()
    valid_pd = valid.as_data_frame()
    test_pd  = test.as_data_frame()

    results = []

    # -------------------------------
    # 1. Baseline GBM
    # -------------------------------
    baseline_params = dict(
        ntrees=100,
        max_depth=5,
        learn_rate=0.1,
        sample_rate=0.9,
        col_sample_rate=0.9,
        balance_classes=True,  # important for imbalanced target
        stopping_rounds=5,
        stopping_metric="AUC",
        seed=176,
    )
    results.append(run_gbm_classification("GBM Baseline", train, valid, test, X, y, baseline_params))

    # -------------------------------
    # 2. Manual GBM Variants
    # -------------------------------
    manual_configs = [
        # ntrees, max_depth, learn_rate, sample_rate, col_sample_rate
        (200, 4, 0.05, 1.0, 1.0),
        (300, 6, 0.05, 0.9, 0.9),
        (500, 8, 0.03, 0.8, 0.8),
        (800, 10, 0.03, 0.7, 0.7),
        (100, 5, 0.1, 1.0, 1.0),
    ]
    for idx, (ntrees, max_depth, lr, samp, col_samp) in enumerate(manual_configs, start=1):
        params = dict(
            ntrees=ntrees,
            max_depth=max_depth,
            learn_rate=lr,
            sample_rate=samp,
            col_sample_rate=col_samp,
            balance_classes=True,
            stopping_rounds=5,
            stopping_metric="AUC",
            seed=176
        )
        results.append(run_gbm_classification(f"GBM Manual {idx}", train, valid, test, X, y, params))

    # -------------------------------
    # 3. GBM with Standardized Features
    # -------------------------------
    numeric_cols = train_pd.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(train_pd[numeric_cols])
    X_valid_std = scaler.transform(valid_pd[numeric_cols])
    X_test_std  = scaler.transform(test_pd[numeric_cols])

    train_std_pd = pd.DataFrame(X_train_std, columns=numeric_cols)
    valid_std_pd = pd.DataFrame(X_valid_std, columns=numeric_cols)
    test_std_pd  = pd.DataFrame(X_test_std, columns=numeric_cols)

    train_std_pd[y] = train_pd[y].values
    valid_std_pd[y] = valid_pd[y].values
    test_std_pd[y]  = test_pd[y].values

    train_std = h2o.H2OFrame(train_std_pd)
    valid_std = h2o.H2OFrame(valid_std_pd)
    test_std  = h2o.H2OFrame(test_std_pd)

    strong_params = dict(
        ntrees=500,
        max_depth=8,
        learn_rate=0.05,
        sample_rate=0.9,
        col_sample_rate=0.9,
        balance_classes=True,
        stopping_rounds=5,
        stopping_metric="AUC",
        seed=176
    )
    results.append(run_gbm_classification("GBM (Standardized Features)", train_std, valid_std, test_std, numeric_cols, y, strong_params))

    # -------------------------------
    # 4. GBM with Gaussianized Features (log1p)
    # -------------------------------
    train_gauss_pd = train_pd.copy()
    valid_gauss_pd = valid_pd.copy()
    test_gauss_pd  = test_pd.copy()

    for col in numeric_cols:
        min_val = train_gauss_pd[col].min()
        shift = 1.0 - min_val if min_val <= 0 else 0.0
        train_gauss_pd[col] = np.log1p(train_gauss_pd[col] + shift)
        valid_gauss_pd[col] = np.log1p(valid_gauss_pd[col] + shift)
        test_gauss_pd[col]  = np.log1p(test_gauss_pd[col] + shift)

    train_gauss_pd[y] = train_pd[y].values
    valid_gauss_pd[y] = valid_pd[y].values
    test_gauss_pd[y]  = test_pd[y].values

    train_gauss = h2o.H2OFrame(train_gauss_pd)
    valid_gauss = h2o.H2OFrame(valid_gauss_pd)
    test_gauss  = h2o.H2OFrame(test_gauss_pd)

    results.append(run_gbm_classification("GBM (Gaussianized Features)", train_gauss, valid_gauss, test_gauss, numeric_cols, y, strong_params))

    # -------------------------------
    # 5. PCA-based GBM (on standardized numeric features)
    # -------------------------------
    pca = PCA(n_components=0.95, random_state=176)
    X_train_pca = pca.fit_transform(X_train_std)
    X_valid_pca = pca.transform(X_valid_std)
    X_test_pca  = pca.transform(X_test_std)

    pc_names = [f"PC{i+1}" for i in range(X_train_pca.shape[1])]

    train_pca_pd = pd.DataFrame(X_train_pca, columns=pc_names)
    valid_pca_pd = pd.DataFrame(X_valid_pca, columns=pc_names)
    test_pca_pd  = pd.DataFrame(X_test_pca, columns=pc_names)

    train_pca_pd[y] = train_pd[y].values
    valid_pca_pd[y] = valid_pd[y].values
    test_pca_pd[y]  = test_pd[y].values

    train_pca = h2o.H2OFrame(train_pca_pd)
    valid_pca = h2o.H2OFrame(valid_pca_pd)
    test_pca  = h2o.H2OFrame(test_pca_pd)

    results.append(run_gbm_classification("GBM (PCA 95% variance)", train_pca, valid_pca, test_pca, pc_names, y, strong_params))

    # -------------------------------
    # 6. Build summary table
    # -------------------------------
    results_df = pd.DataFrame([{k:v for k,v in r.items() if k != "model"} for r in results])
    results_df_sorted = results_df.sort_values("test_AUC", ascending=False).reset_index(drop=True)
    print("\n===== Summary Table =====")
    print(results_df_sorted)

    summary_path = os.path.join(OUTPUT_DIR, "stroke_gbm_summary.csv")
    results_df_sorted.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")

    # -------------------------------
    # 7. Feature importance for best model
    # -------------------------------
    best_model_obj = results[results_df_sorted.index[0]]["model"]
    print("\n===== Feature Importance for Best Model =====")
    varimp_df = best_model_obj.varimp(use_pandas=True)
    print(varimp_df.head(20))

if __name__ == "__main__":
    main()
