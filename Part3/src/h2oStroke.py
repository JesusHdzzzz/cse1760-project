# =========================================
#  H2O GBM: Stroke Prediction Example
# =========================================

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd

# ------------------------------
# Start H2O
# ------------------------------
h2o.init(max_mem_size="4G")

# ------------------------------
# Load dataset
# ------------------------------
data_path = "data/healthcare-dataset-stroke-data.csv"
df = h2o.import_file(data_path)

# ------------------------------
# Basic cleaning
# ------------------------------

# Remove rows with missing target
df = df[~df["stroke"].isna()]

# Convert categorical columns
categoricals = [
    "gender", "ever_married", "work_type", "Residence_type",
    "smoking_status"
]

for col in categoricals:
    df[col] = df[col].asfactor()

# Convert target to factor for classification
df["stroke"] = df["stroke"].asfactor()

# ------------------------------
# Train/test split
# ------------------------------
train, test = df.split_frame(ratios=[0.8], seed=123)

x = [c for c in df.columns if c != "stroke"]
y = "stroke"

# ---------------------------------------------
# Hyperparameter grids to test
# ---------------------------------------------
max_depth_values = [2, 3, 4, 5]
learn_rate_values = [0.1, 0.05]
ntrees_values = [100, 200, 300]

# Storage for results
results = []

# ---------------------------------------------
# Grid search (manual loop)
# ---------------------------------------------
for depth in max_depth_values:
    for lr in learn_rate_values:
        for trees in ntrees_values:

            print(f"\nTraining GBM: depth={depth}, lr={lr}, trees={trees}")

            model = H2OGradientBoostingEstimator(
                max_depth=depth,
                learn_rate=lr,
                ntrees=trees,
                sample_rate=0.9,
                col_sample_rate=0.9,
                seed=123,
                stopping_metric="AUC",
                stopping_rounds=3,
                balance_classes=True,  # optional but helps
            )
            
            model.train(x=x, y=y, training_frame=train, validation_frame=test)
            
            perf = model.model_performance(test)

            # Save metrics
            results.append({
                "max_depth": depth,
                "learn_rate": lr,
                "ntrees": trees,
                "AUC": perf.auc(),
                "LogLoss": perf.logloss(),
                "RMSE": perf.rmse(),
                "F1@opt": perf.F1()[0][1],   # row 0 col 1 = F1 value at best threshold
            })

# ---------------------------------------------
# Convert to Pandas table for easy viewing
# ---------------------------------------------
results_df = pd.DataFrame(results)
print(results_df)

# Save to CSV
results_df.to_csv("hyperparameter_results.csv", index=False)
print("Saved results to hyperparameter_results.csv")