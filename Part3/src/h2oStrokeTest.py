import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings
import time

warnings.filterwarnings('ignore', category=FutureWarning)

start_time = time.perf_counter()
# --- 1. DATA LOADING AND PREPARATION (PANDAS) ---

# Load the original dataset
df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

# Fill in the missing bmi values with the median bmi value
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)

# Drop rows where 'bmi' is missing
#df.dropna(subset=['bmi'], inplace=True)

# Drop the 'Other' gender row
df = df[df['gender'] != 'Other']

# --- 2. H2O SETUP AND DATA LOADING ---

# Initialize H2O cluster
h2o.init(max_mem_size="8G")

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
    seed = 1234
)

features_to_drop = ['ever_married', 'Residence_type']

# Create the reduced list of predictors (X_reduced)
X_reduced = [feature for feature in X if feature not in features_to_drop]

# Check class distribution in each split
print("\n--- Class Distribution in Splits ---")
print("\nTraining Set:")
train_dist = train[y].table()
train_counts = train_dist.as_data_frame()
print(f"No Stroke (0): {int(train_counts.iloc[0, 1])}, Stroke (1): {int(train_counts.iloc[1, 1])}")

print("\nValidation Set:")
valid_dist = valid[y].table()
valid_counts = valid_dist.as_data_frame()
print(f"No Stroke (0): {int(valid_counts.iloc[0, 1])}, Stroke (1): {int(valid_counts.iloc[1, 1])}")

print("\nTest Set:")
test_dist = test[y].table()
test_counts = test_dist.as_data_frame()
print(f"No Stroke (0): {int(test_counts.iloc[0, 1])}, Stroke (1): {int(test_counts.iloc[1, 1])}")

totalnostroke = int(train_counts.iloc[0, 1]) + int(valid_counts.iloc[0, 1]) + int(test_counts.iloc[0, 1])
totalstroke = int(train_counts.iloc[1, 1]) + int(valid_counts.iloc[1, 1]) + int(test_counts.iloc[1, 1])
print(f"Totals: No Stroke (0): {totalnostroke}, Stroke (1): {totalstroke}")

# --- 4. DEFINE HYPERPARAMETER GRID ---

# Define the hyperparameter search space
# FIXED: Added more regularization options and better hyperparameter ranges
hyper_params = {
    'max_depth': [3, 4, 5],              # REDUCED from [6, 7, 8] to prevent overfitting
    'learn_rate': [0.01, 0.03, 0.05],    # Added slower learning rates
    'ntrees': [100, 200, 300],           # Reduced range, will use early stopping
    'sample_rate': [0.5, 0.6, 0.7],      # Added lower values for more regularization
    'col_sample_rate': [0.7, 0.8, 0.9],  # Added lower values for more regularization
    'min_rows': [10, 20]                 # INCREASED from [2] to prevent memorization
}

# --- 5. GBM MODEL TRAINING WITH GRID SEARCH AND BALANCING ---

# Initialize the base GBM model estimator
# FIXED: Reduced class_sampling_factors and added early stopping
gbm_base = H2OGradientBoostingEstimator(
    seed=1234,
    #balance_classes=True,
    class_sampling_factors=[1.0, 6.0],   # FIXED: Reduced from 20.0 to 6.0
    
    # ADDED: Early stopping to prevent overfitting
    stopping_rounds=10,                   # Stop if no improvement for 10 rounds
    stopping_metric="aucpr",              # Monitor AUCPR on validation set
    stopping_tolerance=0.001,             # Minimum improvement threshold
    
    nfolds=5,
    keep_cross_validation_predictions=True,
    fold_assignment="Stratified"
)

# Initialize the Grid Search
grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria={'strategy': "Cartesian"}
)

# Train the Grid Search
print("\nStarting H2O GBM Grid Search with Class Balancing...")
grid.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid
)

grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)

# Extract ntrees and AUCPR for each model
n_trees_list = []
max_depth_list = []
learn_rate_list = []
sample_rate_list = []
col_sample_rate_list = []
min_rows_list = []
train_aucpr_list = []
valid_aucpr_list = []
test_aucpr_list = []

print("\n--- Extracting results from Grid Search ---")
for model in grid_perf.models:
    ntrees = model.actual_params['ntrees']
    max_depth = model.actual_params['max_depth']
    learn_rate = model.actual_params['learn_rate']
    sample_rate = model.actual_params['sample_rate']
    col_sample_rate = model.actual_params['col_sample_rate']
    min_rows = model.actual_params['min_rows']
    
    # Get performance on each dataset
    train_perf = model.model_performance(test_data=train)
    valid_perf = model.model_performance(test_data=valid)
    test_perf = model.model_performance(test_data=test)
    
    n_trees_list.append(ntrees)
    max_depth_list.append(max_depth)
    learn_rate_list.append(learn_rate)
    sample_rate_list.append(sample_rate)
    col_sample_rate_list.append(col_sample_rate)
    min_rows_list.append(min_rows)
    train_aucpr_list.append(train_perf.aucpr())
    valid_aucpr_list.append(valid_perf.aucpr())
    test_aucpr_list.append(test_perf.aucpr())
    
    #print(f"Trees: {ntrees}, Train: {train_perf.aucpr():.4f}, Valid: {valid_perf.aucpr():.4f}, Test: {test_perf.aucpr():.4f}")

# Sort by number of trees for plotting
import matplotlib.pyplot as plt
sorted_data = sorted(zip(n_trees_list, train_aucpr_list, valid_aucpr_list, test_aucpr_list))
n_trees_sorted, train_sorted, valid_sorted, test_sorted = zip(*sorted_data)

# Plot AUCPR vs Number of Trees
plt.figure(figsize=(10, 6))
plt.plot(n_trees_sorted, train_sorted, marker='o', label='Train AUCPR', linewidth=2)
plt.plot(n_trees_sorted, valid_sorted, marker='s', label='Validation AUCPR', linewidth=2)
plt.plot(n_trees_sorted, test_sorted, marker='^', label='Test AUCPR', linewidth=2)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('AUCPR', fontsize=12)
plt.title('AUCPR vs Number of Trees', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('aucpr_vs_ntrees_fixed.png', dpi=300)
print("\nPlot saved as 'aucpr_vs_ntrees_fixed.png'")
#plt.show()

# --- 6. MODEL EVALUATION AND SELECTION ---

# Get the grid results and sort by AUC (Area Under the Curve)
#grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)
print("\n--- Top Models from Grid Search (Sorted by AUCPR) ---")
#print(grid_perf)

# Select the best model based on validation AUCPR
best_gbm = grid_perf.models[0]

# Evaluate the best model on the unseen test set
performance = best_gbm.model_performance(test_data=test)

'''
predictions = best_gbm.predict(test)
pred_df = predictions.as_data_frame()
actual_df = test[y].as_data_frame()

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

y_true = actual_df[y].values
y_scores = pred_df['p1'].values

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
aucpr = average_precision_score(y_true, y_scores)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, 'b-', linewidth=2, label=f'AUCPR = {aucpr:.3f}')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('simple_aucpr.png')
plt.show()
'''

# --- 6. MODEL EVALUATION AND SELECTION (Continued) ---
# Print final results
print(f"\n--- Best Model Hyperparameters ---")
print(f"Max Depth: {best_gbm.actual_params['max_depth']}, Learning Rate: {best_gbm.actual_params['learn_rate']}, Number of Trees: {best_gbm.actual_params['ntrees']}")

# ADDED: Print overfitting analysis for all models
print("\n--- Overfitting Analysis for All Models ---")
count = 0
for model in grid_perf.models:
    performance = model.model_performance(test_data=test)
    
    train_perf = model.model_performance(test_data=train)
    valid_perf = model.model_performance(test_data=valid)
    overfitting_gap = train_aucpr_list[count] - valid_aucpr_list[count]

    print("\n--- Model Evaluation on Test Set ---")
    print(f"Trees: {n_trees_list[count]}")
    print(f"Max Depth: {max_depth_list[count]}")
    print(f"Learn Rate: {learn_rate_list[count]}")
    print(f"Sample Rate: {sample_rate_list[count]}")
    print(f"Col Sample Rate: {col_sample_rate_list[count]}")
    print(f"Min Rows: {min_rows_list[count]}")
    print(f"Train AUCPR: {train_aucpr_list[count]:.4f}")
    print(f"Valid AUCPR: {valid_aucpr_list[count]:.4f}")
    print(f"Test AUCPR: {test_aucpr_list[count]:.4f}")
    print(f"Overfitting Gap (Train-Valid): {overfitting_gap:.4f}")
    count += 1
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

    precision_class0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0
    macro_f1 = (f1 + f1_class0) / 2

    total = tp + tn + fp + fn
    weighted_f1 = (f1_class0 * (tn + fp) + f1 * (tp + fn)) / total

    print("\n--- Metrics at Max F1 Threshold ---")
    print(f"True Positives (Strokes Caught): {tp}")
    print(f"False Negatives (Strokes Missed): {fn}")
    print(f"True Negatives (Correctly predicted no stroke): {tn}")
    print(f"False Positives (False alarms): {fp}")
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# --- 7. FEATURE SELECTION AND FINAL MODEL TRAINING ---

# 7.1. Identify Features to Keep (Filtering based on Importance > 0)

# Create a list of features to drop (those with 0 importance from previous output)
features_to_drop = ['ever_married', 'Residence_type']

# Create the reduced list of predictors (X_reduced)
X_reduced = [feature for feature in X if feature not in features_to_drop]

print(f"\nFeatures dropped based on 0 importance: {features_to_drop}")
print(f"Features kept for final model: {X_reduced}")

# 7.2. Extract Best Hyperparameters

# Best parameters found in grid search were:
best_depth = best_gbm.actual_params['max_depth']
best_rate = best_gbm.actual_params['learn_rate']
best_trees = best_gbm.actual_params['ntrees']
best_sample = best_gbm.actual_params['sample_rate']
best_col = best_gbm.actual_params['col_sample_rate']
best_row = best_gbm.actual_params['min_rows']

# 7.3. Train Final Model on Reduced Feature Set

print("\nTraining Final GBM Model with Reduced Feature Set...")

final_gbm = H2OGradientBoostingEstimator(
    # Use the best hyperparameters
    ntrees=best_trees,
    max_depth=best_depth,
    learn_rate=best_rate,
    sample_rate=best_sample,
    col_sample_rate=best_col,
    min_rows=best_row,
    seed=1234,
    #balance_classes=True,
    class_sampling_factors=[1.0, 6.0],    # FIXED: Reduced from 20.0 to 6.0
    
    # ADDED: Early stopping
    stopping_rounds=10,
    stopping_metric="aucpr",
    stopping_tolerance=0.001,
    
    nfolds=5,
    keep_cross_validation_predictions=True,
    fold_assignment="Stratified"
)

# Train using the reduced feature set
final_gbm.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid
)

# 7.4. Evaluate Reduced Model (If using full train set, skip test evaluation)
# To maintain evaluation consistency, we'll evaluate the reduced model on the 'test' split.
final_performance = final_gbm.model_performance(test_data=test)

print("\n--- Final Model Evaluation (Reduced Features) ---")
print(f"Reduced Model Test Set AUC: {final_performance.auc()}")
print(f"Reduced Model Test Set AUCPR: {final_performance.aucpr()}")

# ADDED: Print generalization metrics
final_train_perf = final_gbm.model_performance(test_data=train)
final_valid_perf = final_gbm.model_performance(test_data=valid)
print(f"\nGeneralization Analysis:")
print(f"Train AUCPR: {final_train_perf.aucpr():.4f}")
print(f"Valid AUCPR: {final_valid_perf.aucpr():.4f}")
print(f"Test AUCPR: {final_performance.aucpr():.4f}")
print(f"Overfitting Gap (Train-Valid): {final_train_perf.aucpr() - final_valid_perf.aucpr():.4f}")

# --- FINAL MODEL EVALUATION (REVISED BLOCK) ---

print("\n--- Final Model Classification Metrics ---")

# Get the confusion matrix first
cm = final_performance.confusion_matrix()
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

precision_class0 = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_class0 = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0) if (precision_class0 + recall_class0) > 0 else 0
macro_f1 = (f1 + f1_class0) / 2

total = tp + tn + fp + fn
weighted_f1 = (f1_class0 * (tn + fp) + f1 * (tp + fn)) / total

print("\n--- Metrics at Max F1 Threshold ---")
print(f"True Positives (Strokes Caught): {tp}")
print(f"False Negatives (Strokes Missed): {fn}")
print(f"True Negatives (Correctly predicted no stroke): {tn}")
print(f"False Positives (False alarms): {fp}")
print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score: {f1:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Get standard metrics
print("\n--- Performance Metrics ---")
print(f"AUC: {final_performance.auc()}")
print(f"AUCPR: {final_performance.aucpr()}")

end_time = time.perf_counter()
print("\ntime taken: ", (end_time - start_time))


# --- 8. SHUTDOWN ---

h2o.cluster().shutdown()