import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. DATA LOADING AND PREPARATION (PANDAS) ---

# Load the original dataset
df = pd.read_csv("data/healthcare-dataset-stroke-data.csv")

# Fill in the missing bmi values with the median bmi value
#bmi_median = df['bmi'].median()
#df['bmi'].fillna(bmi_median, inplace=True)

# Drop rows where 'bmi' is missing
#df.dropna(subset=['bmi'], inplace=True)

# Drop the 'Other' gender row
#df = df[df['gender'] != 'Other']

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
    ratios=[0.7, 0.15],
    seed=42
)

# --- 4. DEFINE HYPERPARAMETER GRID ---

# Define the hyperparameter search space
hyper_params = {
    'max_depth': [5],
    'learn_rate': [0.05],
    'ntrees': [300]
}

# --- 5. GBM MODEL TRAINING WITH GRID SEARCH AND BALANCING ---

# Initialize the base GBM model estimator
gbm_base = H2OGradientBoostingEstimator(
    seed=42,
    balance_classes=True,
    stopping_rounds=0,
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

# --- 6. MODEL EVALUATION AND SELECTION (Continued) ---

# Get the Feature Importance for the best model
importance_df = best_gbm.varimp(use_pandas=True)

print("\n--- Feature Importance (Top Predictors) ---")
# Display the top features and their relative importance
print(importance_df.head(10).to_markdown(index=False, numalign="left", stralign="left"))

# Print final results
print(f"\n--- Best Model Hyperparameters ---")
print(f"Max Depth: {best_gbm.actual_params['max_depth']}, Learning Rate: {best_gbm.actual_params['learn_rate']}, Number of Trees: {best_gbm.actual_params['ntrees']}")

print("\n--- Best Model Evaluation on Test Set ---")
print(f"Test Set AUC: {performance.auc()}")
print(f"Test Set AUCPR: {performance.aucpr()}")
#print(f"Test Set Gini: {performance.gini()}")
#print(f"Test Set F1 Score: {performance.f1()}")

cm = performance.confusion_matrix()
print("\n--- Confusion Matrix ---")
print(cm)

cm_table = cm.to_list()
tn = cm_table[0][0]  # True Negatives
fp = cm_table[0][1]  # False Positives
fn = cm_table[1][0]  # False Negatives
tp = cm_table[1][1]  # True Positives

# Calculate metrics manually
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n--- Metrics at Max F1 Threshold ---")
print(f"True Positives (Strokes Caught): {tp}")
print(f"False Negatives (Strokes Missed): {fn}")
print(f"True Negatives (Correctly predicted no stroke): {tn}")
print(f"False Positives (False alarms): {fp}")
print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- 7. FEATURE SELECTION AND FINAL MODEL TRAINING ---
'''
# 7.1. Identify Features to Keep (Filtering based on Importance > 0)

# Create a list of features to drop (those with 0 importance from previous output)
features_to_drop = ['heart_disease', 'ever_married', 'Residence_type']

# Create the reduced list of predictors (X_reduced)
X_reduced = [feature for feature in X if feature not in features_to_drop]

print(f"\nFeatures dropped based on 0 importance: {features_to_drop}")
print(f"Features kept for final model: {X_reduced}")

# 7.2. Extract Best Hyperparameters

# Best parameters found in grid search were:
best_depth = best_gbm.actual_params['max_depth']
best_rate = best_gbm.actual_params['learn_rate']
best_trees = best_gbm.actual_params['ntrees']

# 7.3. Train Final Model on Reduced Feature Set

print("\nTraining Final GBM Model with Reduced Feature Set...")

final_gbm = H2OGradientBoostingEstimator(
    # Use the best hyperparameters
    ntrees=best_trees,
    max_depth=best_depth,
    learn_rate=best_rate,
    # Use the same stabilization parameters
    seed=42,
    balance_classes=True,
    model_id="final_stroke_gbm_reduced"
)

# Train using the reduced feature set
final_gbm.train(
    x=X_reduced,
    y=y,
    training_frame=train
)

# 7.4. Evaluate Reduced Model (If using full train set, skip test evaluation)
# To maintain evaluation consistency, we'll evaluate the reduced model on the 'test' split.
final_performance = final_gbm.model_performance(test_data=test)

print("\n--- Final Model Evaluation (Reduced Features) ---")
print(f"Reduced Model Test Set AUC: {final_performance.auc()}")
print(f"Reduced Model Test Set AUCPR: {final_performance.aucpr()}")
print(f"Reduced Model Test Set Gini: {final_performance.gini()}")

# --- FINAL MODEL EVALUATION (REVISED BLOCK) ---

print("\n--- Final Model Classification Metrics ---")

# Get the confusion matrix first
cm = final_performance.confusion_matrix()
print("\n--- Confusion Matrix ---")
print(cm)

# Get standard metrics
print("\n--- Performance Metrics ---")
print(f"AUC: {final_performance.auc()}")
print(f"Gini: {final_performance.gini()}")
print(f"Logloss: {final_performance.logloss()}")

# Get F1 scores at different thresholds
print("\n--- F1 Score ---")
f1_scores = final_performance.F1()
print(f1_scores)

# Get precision at different thresholds
print("\n--- Precision ---")
precision_scores = final_performance.precision()
print(precision_scores)

# Get recall at different thresholds
print("\n--- Recall (Sensitivity) ---")
recall_scores = final_performance.recall()
print(recall_scores)

# Get accuracy
print("\n--- Accuracy ---")
accuracy_scores = final_performance.accuracy()
print(accuracy_scores)

# If you want the metrics at the default threshold (usually 0.5), 
# these methods return arrays, so you can access specific threshold values
print("\n--- Summary at Default Threshold ---")
# The arrays are indexed by threshold, you may need to find the right index
# or simply display the entire output which shows metrics across thresholds
'''
# --- 8. SHUTDOWN ---

h2o.cluster().shutdown()