import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. DATA LOADING AND PREPARATION (PANDAS) ---

# Load the original dataset
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# --- 2. H2O SETUP AND DATA LOADING ---

# Initialize H2O cluster
h2o.init(nthreads=-1)

# Convert the cleaned pandas DataFrame to an H2O Frame
h2o_df = h2o.H2OFrame(df)

# Define predictor and target variables
y = 'stroke'
X = h2o_df.columns
X.remove('id')
X.remove(y)

# Convert the target variable to a factor (essential for classification)
h2o_df[y] = h2o_df[y].asfactor()

# --- 3. DATA SPLITTING ---

# Split the H2O Frame into train, validation, and test sets
train, valid, test = h2o_df.split_frame(
    ratios=[0.7, 0.15],  # 70% train, 15% validation, 15% test
    seed=42
)

# --- 4. DEFINE HYPERPARAMETER GRID ---

# Define the hyperparameter search space
hyper_params = {
    'max_depth': [3, 5, 8],
    'learn_rate': [0.01, 0.05, 0.1],
    'ntrees': [100, 200]
}

# --- 5. GBM MODEL TRAINING WITH GRID SEARCH AND BALANCING ---

# Initialize the base GBM model estimator
gbm_base = H2OGradientBoostingEstimator(
    seed=42,
    balance_classes=True,
    stopping_rounds=5,
    stopping_metric="AUC",
    score_tree_interval=1
)

# Initialize the Grid Search
grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria={'strategy': "Cartesian"} # Search all combinations
)

# Train the Grid Search
print("\nStarting H2O GBM Grid Search with Class Balancing...")
grid.train(
    x=X,
    y=y,
    training_frame=train,
    validation_frame=valid
)

# --- 6. MODEL EVALUATION AND SELECTION ---

# Get the grid results and sort by AUC (Area Under the Curve)
grid_perf = grid.get_grid(sort_by="auc", decreasing=True)
print("\n--- Top Models from Grid Search (Sorted by AUC) ---")
print(grid_perf)

# Select the best model based on validation AUC
best_gbm = grid_perf.models[0]

# Evaluate the best model on the unseen test set
performance = best_gbm.model_performance(test_data=test)

# Print final results
print(f"\n--- Best Model Hyperparameters ---")
print(f"Max Depth: {best_gbm.actual_params['max_depth']}, Learning Rate: {best_gbm.actual_params['learn_rate']}, Number of Trees: {best_gbm.actual_params['ntrees']}")

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set AUC: {performance.auc()}")
print(f"Test Set Gini: {performance.gini()}")
print(f"Test Set F1 Score: {performance.f1()}")

# --- 7. SHUTDOWN ---

# Shut down the H2O cluster when finished
h2o.cluster().shutdown()