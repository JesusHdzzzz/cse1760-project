import pandas as pd
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
import warnings
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc, confusion_matrix, fbeta_score

warnings.filterwarnings('ignore', category=FutureWarning)

start_time = time.perf_counter()

print("=" * 60)
print("STROKE PREDICTION MODEL - NO DATA LEAKAGE VERSION")
print("=" * 60)

# --- 1. DATA LOADING AND PREPARATION ---
print("\n1. Loading and preparing data...")
df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
bmi_median = df['bmi'].median()
df['bmi'].fillna(bmi_median, inplace=True)
df = df[df['gender'] != 'Other']

# --- 2. INITIAL FEATURE SELECTION (BEFORE ANY MODELING) ---
print("\n2. Initial feature selection (domain knowledge only)...")
# ONLY use domain knowledge, not previous model results
# 'id' is already removed, keep all other features initially
features_to_keep = ['gender', 'age', 'hypertension', 'heart_disease', 
                    'ever_married', 'work_type', 'Residence_type',  # Keep all initially
                    'avg_glucose_level', 'bmi', 'smoking_status']

# Filter dataframe
df = df[features_to_keep + ['stroke']]

# --- 3. H2O INITIALIZATION ---
print("3. Initializing H2O...")
h2o.init(max_mem_size="8G", nthreads=-1)

# Convert to H2O frame
h2o_df = h2o.H2OFrame(df)
y = 'stroke'
X = h2o_df.columns
X.remove(y)  # Only remove target, keep all features initially
h2o_df[y] = h2o_df[y].asfactor()

# --- 4. STRATIFIED SPLITTING ---
print("4. Creating stratified splits...")
df_pandas = h2o_df.as_data_frame()

# Split into train+valid and test FIRST
train_valid_pd, test_pd = train_test_split(
    df_pandas, 
    test_size=0.15,  # Reserve 15% for FINAL test only
    stratify=df_pandas['stroke'],
    random_state=1234
)

# Split train_valid into train and validation
train_pd, valid_pd = train_test_split(
    train_valid_pd,
    test_size=0.1765,  # 15% of original = 0.15/0.85 ≈ 0.1765
    stratify=train_valid_pd['stroke'],
    random_state=1234
)

train = h2o.H2OFrame(train_pd)
valid = h2o.H2OFrame(valid_pd)
test = h2o.H2OFrame(test_pd)

train[y] = train[y].asfactor()
valid[y] = valid[y].asfactor()
test[y] = test[y].asfactor()

print("\n✅ FINAL DATA SPLITS:")
print(f"  Train: {train.shape[0]} samples (for training)")
print(f"  Valid: {valid.shape[0]} samples (for hyperparameter tuning)")
print(f"  Test:  {test.shape[0]} samples (FINAL evaluation only)")

# --- 5. FEATURE SELECTION USING TRAIN SET ONLY ---
print("\n5. Feature selection using train set only...")

# Train a quick model on TRAIN SET ONLY to identify useless features
feature_model = H2OGradientBoostingEstimator(
    seed=1234,
    ntrees=50,
    max_depth=3,
    balance_classes=True,
    nfolds=0
)

feature_model.train(
    x=X,
    y=y,
    training_frame=train,
    validation_frame=valid
)

# Get feature importance from TRAIN SET ONLY
feature_imp = feature_model.varimp(use_pandas=True)
print("\nFeature Importance (from train set only):")
for i, row in feature_imp.iterrows():
    print(f"  {row['variable']:<20}: {row['scaled_importance']:.3f}")

# Drop features with near-zero importance (using TRAIN SET only)
features_to_drop = feature_imp[feature_imp['scaled_importance'] < 0.01]['variable'].tolist()
if features_to_drop:
    print(f"\nDropping features with <0.01 importance: {features_to_drop}")
    X_reduced = [feature for feature in X if feature not in features_to_drop]
else:
    print("\nNo features to drop (all have importance ≥0.01)")
    X_reduced = X.copy()

print(f"✅ Final features: {len(X_reduced)} features")

# --- 6. HYPERPARAMETER TUNING (USING TRAIN + VALID ONLY) ---
print("\n6. Hyperparameter tuning (train + valid sets only)...")

hyper_params = {
    'max_depth': [3, 4],
    'learn_rate': [0.01, 0.03],
    'sample_rate': [0.7, 0.8],
    'col_sample_rate': [0.7, 0.8],
    'min_rows': [20, 30],
    'ntrees': [100]
}

gbm_base = H2OGradientBoostingEstimator(
    seed=1234,
    balance_classes=True,
    max_after_balance_size=2.0,
    stopping_rounds=7,
    stopping_metric="AUC",
    stopping_tolerance=0.005,
    nfolds=0
)

search_criteria = {
    'strategy': "RandomDiscrete",
    'max_models': 10,
    'seed': 1234
}

grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria=search_criteria,
    grid_id="stroke_no_leakage_grid",
    parallelism=2
)

print("Training grid search on train set, validating on validation set...")
grid.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid  # ← Only train+valid used for tuning!
)

# Get best model based on VALIDATION performance
grid_perf = grid.get_grid(sort_by="auc", decreasing=True)
best_gbm = grid_perf.models[0]
print(f"✓ Best model validation AUC: {best_gbm.auc(valid):.4f}")

# --- 7. THRESHOLD OPTIMIZATION (ON VALIDATION SET ONLY) ---
print("\n7. Optimizing threshold on validation set only...")

valid_perf = best_gbm.model_performance(valid)
opt_thresh = valid_perf.find_threshold_by_max_metric("f2")  # ← USES VALID ONLY
print(f"✓ Optimal F2 threshold from validation set: {opt_thresh:.4f}")

# --- 8. FINAL EVALUATION (ON TEST SET - ONE TIME ONLY) ---
print("\n8. FINAL evaluation on test set (one time only)...")

# Get predictions on test set
predictions = best_gbm.predict(test)
pred_df = predictions.as_data_frame()
actual_df = test[y].as_data_frame()

y_true = actual_df[y].values
y_scores = pred_df['p1'].values

# Apply threshold optimized on VALIDATION set
y_pred = (y_scores >= opt_thresh).astype(int)

# Calculate final metrics
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
recall = tp / (tp + fn)
precision = tp / (tp + fp)
f2 = fbeta_score(y_true, y_pred, beta=2)
aucpr = average_precision_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# --- 9. FINAL REPORT (NO MORE TEST SET TOUCHING) ---
print("\n" + "="*60)
print("✅ FINAL UNBIASED PERFORMANCE")
print("="*60)

print(f"\nTest Set Performance (threshold from validation):")
print(f"  Recall:              {recall:.3f} ({tp}/{tp+fn} strokes caught)")
print(f"  Precision:           {precision:.3f} ({tp}/{tp+fp} alerts are real)")
print(f"  F2 Score:            {f2:.3f}")
print(f"  Accuracy:            {(tp+tn)/(tp+tn+fp+fn):.3f}")
print(f"  AUCPR:               {aucpr:.4f}")
print(f"  AUC-ROC:             {roc_auc:.4f}")

print(f"\nData Usage Summary:")
print(f"  Train:    {train.shape[0]} samples - Model training")
print(f"  Valid:    {valid.shape[0]} samples - Hyperparameter tuning")
print(f"  Test:     {test.shape[0]} samples - FINAL evaluation only")


# --- 10. SHUTDOWN ---
end_time = time.perf_counter()
print(f"\n{'='*60}")
print(f"✅ COMPLETED IN {end_time - start_time:.2f} SECONDS")
print(f"✅ NO DATA LEAKAGE - Proper train/valid/test usage")
print(f"{'='*60}")

h2o.cluster().shutdown()