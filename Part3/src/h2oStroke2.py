import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix, fbeta_score

warnings.filterwarnings('ignore', category=FutureWarning)

start_time = time.perf_counter()

print("=" * 60)
print("STROKE PREDICTION MODEL - SIMPLIFIED WORKING VERSION")
print("=" * 60)

# --- 1. DATA LOADING AND PREPARATION ---
print("\n1. Loading and preparing data...")
df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)
df = df[df['gender'] != 'Other']

# --- 2. H2O INITIALIZATION ---
print("2. Initializing H2O...")
h2o.init(max_mem_size="8G", nthreads=-1)

# Convert to H2O frame
h2o_df = h2o.H2OFrame(df)
y = 'stroke'
X = h2o_df.columns
X.remove('id')
X.remove(y)
h2o_df[y] = h2o_df[y].asfactor()

# --- 3. STRATIFIED SPLITTING ---
print("3. Creating stratified splits...")
df_pandas = h2o_df.as_data_frame()
train_pd, temp_pd = train_test_split(
    df_pandas, 
    test_size=0.3, 
    stratify=df_pandas['stroke'],
    random_state=1234
)
valid_pd, test_pd = train_test_split(
    temp_pd, 
    test_size=0.5, 
    stratify=temp_pd['stroke'],
    random_state=1234
)

train = h2o.H2OFrame(train_pd)
valid = h2o.H2OFrame(valid_pd)
test = h2o.H2OFrame(test_pd)

train[y] = train[y].asfactor()
valid[y] = valid[y].asfactor()
test[y] = test[y].asfactor()

# Show class distribution
print("\n✅ Class Distribution:")
for name, frame in [("Train", train), ("Valid", valid), ("Test", test)]:
    tab = frame[y].table().as_data_frame()
    n0, n1 = int(tab.iloc[0,1]), int(tab.iloc[1,1])
    print(f"  {name}: {n0} non-stroke, {n1} stroke ({n1/(n0+n1)*100:.1f}%)")

# --- 4. FEATURE SELECTION ---
features_to_drop = ['ever_married', 'Residence_type']
X_reduced = [feature for feature in X if feature not in features_to_drop]
print(f"\n✅ Features: {len(X_reduced)} features kept")

# --- 5. TRAIN A SINGLE MODEL FIRST (FOR DEBUGGING) ---
print("\n4. Testing with a single model first...")
single_model = H2OGradientBoostingEstimator(
    seed=1234,
    balance_classes=True,
    max_after_balance_size=3.0,
    ntrees=100,
    max_depth=4,
    learn_rate=0.01,
    min_rows=20,
    sample_rate=0.7,
    col_sample_rate=0.7,
    stopping_rounds=10,
    stopping_metric="AUCPR",
    stopping_tolerance=0.001
)

single_model.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid
)

print(f"✓ Single model validation AUCPR: {single_model.aucpr(valid):.4f}")

# --- 6. GRID SEARCH (SIMPLIFIED) ---
print("\n5. Starting grid search...")

# Small hyperparameter grid for testing
hyper_params = {
    'max_depth': [3, 4, 5],
    'learn_rate': [0.01, 0.02],
    'min_rows': [15, 25],
    'sample_rate': [0.7],
    'col_sample_rate': [0.7]
}

# Base model for grid search
gbm_base = H2OGradientBoostingEstimator(
    seed=1234,
    balance_classes=True,
    max_after_balance_size=3.0,
    ntrees=100,
    stopping_rounds=10,
    stopping_metric="AUCPR",
    stopping_tolerance=0.001
)

# SIMPLE search criteria - just Cartesian
search_criteria = {
    'strategy': "RandomDiscrete",
    'max_models': 8,  # Limit to 8 random combinations
    'seed': 1234,     # For reproducibility
    'max_runtime_secs': 0  # No time limit (0 means unlimited)
}
# Create grid
grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria=search_criteria
)

# Train grid
grid.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid
)

# Get best model
grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)
best_gbm = grid_perf.models[0]
print(f"✓ Grid search complete! Best validation AUCPR: {best_gbm.aucpr(valid):.4f}")

# --- 7. TEST SET EVALUATION ---
print("\n6. Evaluating on test set...")
test_perf = best_gbm.model_performance(test)
print(f"✓ Test AUCPR: {test_perf.aucpr():.4f}")
print(f"✓ Test AUC: {test_perf.auc():.4f}")

# --- 8. OPTIMAL THRESHOLD ---
print("\n7. Finding optimal threshold...")
opt_thresh = test_perf.find_threshold_by_max_metric("f2")
print(f"✓ Optimal F2 threshold: {opt_thresh:.4f}")

# Get predictions
predictions = best_gbm.predict(test)
pred_df = predictions.as_data_frame()
actual_df = test[y].as_data_frame()

y_true = actual_df[y].values
y_scores = pred_df['p1'].values
y_pred = (y_scores >= opt_thresh).astype(int)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f2 = fbeta_score(y_true, y_pred, beta=2)

print("\n✅ FINAL PERFORMANCE:")
print(f"   Recall:     {recall:.3f} ({tp}/{tp+fn} strokes caught)")
print(f"   Precision:  {precision:.3f} ({tp}/{tp+fp} alerts are real)")
print(f"   F2 Score:   {f2:.3f}")
print(f"   Accuracy:   {(tp+tn)/(tp+tn+fp+fn):.3f}")

# --- 9. SHUTDOWN ---
end_time = time.perf_counter()
print(f"\n{'='*60}")
print(f"✅ COMPLETED IN {end_time - start_time:.2f} SECONDS")
print(f"{'='*60}")

h2o.cluster().shutdown()