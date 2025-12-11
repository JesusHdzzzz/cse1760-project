# =========================================
#  H2O GBM: Stroke Prediction Example
# =========================================

import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
train, validation, test = df.split_frame(ratios=[0.7, 0.15], seed=123)

x = [c for c in df.columns if c != "stroke"]
y = "stroke"
'''
hyper_params = {
    "max_depth": [3, 5, 8],
    "learn_rate": [0.01, 0.05, 0.1],
    "ntrees": [100, 200, 300]
}

grid = H2OGridSearch(
    model=H2OGradientBoostingEstimator(
        sample_rate=0.9,
        col_sample_rate=0.9,
        nfolds=5,
        balance_classes=True,
        seed=1234
    ),
    hyper_params=hyper_params,
    grid_id="gbm_stroke_grid"
)

grid.train(x=x, y=y, training_frame=train)
# ---------------------------------------------
# Convert to Pandas table for easy viewing
# ---------------------------------------------
results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(8,6))
for lr in results_df["learn_rate"].unique():
    subset = results_df[results_df["learn_rate"] == lr]
    plt.plot(subset["max_depth"], subset["AUC"], marker="o", label=f"lr={lr}")

plt.title("AUC vs Max Depth for Different Learning Rates")
plt.xlabel("Max Depth")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.show()
'''

# ------------------------------
# Build GBM model
# ------------------------------
gbm = H2OGradientBoostingEstimator(
    ntrees=300,
    max_depth=5,
    learn_rate=0.05,
    nfolds=5,
    keep_cross_validation_models=True,
    keep_cross_validation_predictions=True,
    seed=1234,
    stopping_rounds=5,
    stopping_metric="AUC",
    stopping_tolerance=0.001,
    balance_classes = True
)

gbm.train(x=x, y=y, training_frame=train)

# ------------------------------
# Performance evaluation
# ------------------------------
print("Model Performance (Test Set):")
perf = gbm.model_performance(test)
print(perf)

# Show AUC
print("AUC:", perf.auc())

# ------------------------------
# Variable importance
# ------------------------------
print("Variable Importance:")
print(gbm.varimp(True))


# Save to CSV
#results_df.to_csv("hyperparameter_results.csv", index=False)
#print("Saved results to hyperparameter_results.csv")