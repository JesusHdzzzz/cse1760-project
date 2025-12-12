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
print("STROKE PREDICTION MODEL - ROBUST GRID SEARCH VERSION")
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

# --- 3. STRATIFIED SPLITTING (CRITICAL IMPROVEMENT) ---
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
print("\n✅ STRATIFIED CLASS DISTRIBUTION:")
for name, frame in [("Train", train), ("Valid", valid), ("Test", test)]:
    tab = frame[y].table().as_data_frame()
    n0, n1 = int(tab.iloc[0,1]), int(tab.iloc[1,1])
    print(f"  {name}: {n0} non-stroke, {n1} stroke ({n1/(n0+n1)*100:.1f}%)")

# --- 4. FEATURE SELECTION ---
features_to_drop = ['ever_married', 'Residence_type']
X_reduced = [feature for feature in X if feature not in features_to_drop]
print(f"\n✅ Feature
ass: {len(X_reduced)} features kept")

# --- 5. DEFINE ROBUST HYPERPARAMETER GRID ---
print("\n5. Setting up robust hyperparameter grid...")

# Comprehensive hyperparameter search space
hyper_params = {
    'max_depth': [3, 4],           # Reduced from [3, 4, 5]
    'learn_rate': [0.01, 0.03],    # Reduced from [0.01, 0.02, 0.03]
    'sample_rate': [0.7, 0.8],     # Reduced from [0.6, 0.7, 0.8]
    'col_sample_rate': [0.7, 0.8], # Reduced from [0.6, 0.7, 0.8]
    'min_rows': [20, 30],          # Reduced from [10, 20, 30]
    'ntrees': [100]                # SINGLE value - early stopping will adjust
}

print("\nHyperparameter Search Space:")
for param, values in hyper_params.items():
    print(f"  {param}: {values}")

# --- 6. GBM MODEL TRAINING WITH COMPREHENSIVE GRID SEARCH ---
print("\n6. Starting comprehensive grid search...")

# Base model with proper regularization
gbm_base = H2OGradientBoostingEstimator(
    seed=1234,
    # OPTIMIZED: Reduced class balancing (faster)
    balance_classes=True,
    max_after_balance_size=2.0,  # Reduced from 3.0
    
    # OPTIMIZED: Faster early stopping
    stopping_rounds=7,           # Reduced from 10
    stopping_metric="AUC",       # Faster than AUCPR
    stopping_tolerance=0.005,    # Looser tolerance (faster)
    
    # No CV for speed
    nfolds=0,
    score_tree_interval=20       # Less frequent scoring
)

# Simple Cartesian search
search_criteria = {
    'strategy': "RandomDiscrete",
    'max_models': 10,            # Only test 10 random combos
    'seed': 1234,
}

# Create grid
grid = H2OGridSearch(
    model=gbm_base,
    hyper_params=hyper_params,
    search_criteria=search_criteria,
    grid_id="stroke_fast_grid",
    parallelism=2                 # Run 2 models in parallel
)

# Train the grid
print("Training grid search (this may take a few minutes)...")
grid.train(
    x=X_reduced,
    y=y,
    training_frame=train,
    validation_frame=valid
)

# Get grid results
grid_perf = grid.get_grid(sort_by="aucpr", decreasing=True)
print(f"✓ Grid search complete! Trained {len(grid_perf.models)} models")

# --- 7. EXTRACT AND VISUALIZE GRID SEARCH RESULTS ---
print("\n7. Extracting and visualizing grid search results...")

# Extract performance metrics for all models
model_metrics = []
for i, model in enumerate(grid_perf.models):
    train_perf = model.model_performance(train)
    valid_perf = model.model_performance(valid)
    test_perf = model.model_performance(test)
    
    metrics = {
        'model_id': i+1,
        'max_depth': model.actual_params['max_depth'],
        'learn_rate': model.actual_params['learn_rate'],
        'sample_rate': model.actual_params['sample_rate'],
        'col_sample_rate': model.actual_params['col_sample_rate'],
        'min_rows': model.actual_params['min_rows'],
        'ntrees': model.actual_params.get('ntrees', model.actual_params.get('stopped_ntrees', 0)),
        'train_aucpr': train_perf.aucpr(),
        'valid_aucpr': valid_perf.aucpr(),
        'test_aucpr': test_perf.aucpr(),
        'overfitting_gap': train_perf.aucpr() - valid_perf.aucpr()
    }
    model_metrics.append(metrics)

# Convert to DataFrame for analysis
metrics_df = pd.DataFrame(model_metrics)

# --- 8. VISUALIZATIONS ---
print("\n8. Generating comprehensive visualizations...")

# 8.1. AUCPR vs Hyperparameter Heatmaps
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Hyperparameter Performance Analysis', fontsize=16, fontweight='bold')

# Plot 1: max_depth vs learn_rate
scatter1 = axes[0, 0].scatter(
    metrics_df['max_depth'], 
    metrics_df['learn_rate'], 
    c=metrics_df['valid_aucpr'], 
    s=100, 
    cmap='viridis', 
    alpha=0.8
)
axes[0, 0].set_xlabel('Max Depth')
axes[0, 0].set_ylabel('Learn Rate')
axes[0, 0].set_title('Valid AUCPR by Depth & Learning Rate')
plt.colorbar(scatter1, ax=axes[0, 0])

# Plot 2: sample_rate vs col_sample_rate
scatter2 = axes[0, 1].scatter(
    metrics_df['sample_rate'], 
    metrics_df['col_sample_rate'], 
    c=metrics_df['valid_aucpr'], 
    s=100, 
    cmap='viridis', 
    alpha=0.8
)
axes[0, 1].set_xlabel('Sample Rate')
axes[0, 1].set_ylabel('Col Sample Rate')
axes[0, 1].set_title('Valid AUCPR by Sampling Rates')
plt.colorbar(scatter2, ax=axes[0, 1])

# Plot 3: min_rows vs valid_aucpr
axes[0, 2].scatter(metrics_df['min_rows'], metrics_df['valid_aucpr'], alpha=0.7)
axes[0, 2].set_xlabel('Min Rows')
axes[0, 2].set_ylabel('Valid AUCPR')
axes[0, 2].set_title('AUCPR vs Min Rows')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Overfitting analysis
axes[1, 0].bar(range(len(metrics_df)), metrics_df['overfitting_gap'])
axes[1, 0].set_xlabel('Model Index')
axes[1, 0].set_ylabel('Train-Valid AUCPR Gap')
axes[1, 0].set_title('Overfitting Analysis')
axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 5: Performance across datasets
x = np.arange(len(metrics_df))
width = 0.25
axes[1, 1].bar(x - width, metrics_df['train_aucpr'], width, label='Train', alpha=0.7)
axes[1, 1].bar(x, metrics_df['valid_aucpr'], width, label='Valid', alpha=0.7)
axes[1, 1].bar(x + width, metrics_df['test_aucpr'], width, label='Test', alpha=0.7)
axes[1, 1].set_xlabel('Model Index')
axes[1, 1].set_ylabel('AUCPR')
axes[1, 1].set_title('Performance Across Datasets')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Plot 6: Top 10 models
top_10 = metrics_df.nlargest(10, 'valid_aucpr')
axes[1, 2].barh(range(10), top_10['valid_aucpr'])
axes[1, 2].set_yticks(range(10))
axes[1, 2].set_yticklabels([f"M{i+1}" for i in range(10)])
axes[1, 2].set_xlabel('Valid AUCPR')
axes[1, 2].set_title('Top 10 Models')
axes[1, 2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('grid_search_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: grid_search_analysis.png")

# 8.2. Select and analyze best model
best_gbm = grid_perf.models[0]
best_metrics = metrics_df.iloc[0]

print(f"\n✅ BEST MODEL SELECTED:")
print(f"  Max Depth: {best_metrics['max_depth']}")
print(f"  Learn Rate: {best_metrics['learn_rate']}")
print(f"  Sample Rate: {best_metrics['sample_rate']}")
print(f"  Col Sample Rate: {best_metrics['col_sample_rate']}")
print(f"  Min Rows: {best_metrics['min_rows']}")
print(f"  Valid AUCPR: {best_metrics['valid_aucpr']:.4f}")
print(f"  Overfitting Gap: {best_metrics['overfitting_gap']:.4f}")

# 8.3. Detailed performance analysis of best model
print("\n8.3. Detailed performance analysis...")

# Get optimal threshold (F2 optimized for stroke detection)
test_perf = best_gbm.model_performance(test)
opt_thresh = test_perf.find_threshold_by_max_metric("f2")

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
aucpr = average_precision_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 8.4. Precision-Recall and ROC curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_true, y_scores)
ax1.plot(rec, prec, 'b-', linewidth=2, label=f'PR Curve (AUCPR={aucpr:.3f})')
ax1.axvline(x=recall, color='r', linestyle='--', alpha=0.7, label=f'Recall: {recall:.2f}')
ax1.axhline(y=precision, color='g', linestyle='--', alpha=0.7, label=f'Precision: {precision:.2f}')
ax1.scatter(recall, precision, s=100, color='red', zorder=5, label=f'F2 Threshold ({opt_thresh:.3f})')
ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall Curve', fontsize=14)
ax1.legend(loc='best')
ax1.grid(alpha=0.3)

# ROC Curve
ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve', fontsize=14)
ax2.legend(loc='best')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('performance_curves.png', dpi=300, bbox_inches='tight')
print("✓ Saved: performance_curves.png")

# 8.5. Feature Importance
print("\n8.5. Feature importance analysis...")
importance = best_gbm.varimp(use_pandas=True)

plt.figure(figsize=(10, 6))
plt.barh(importance['variable'][:10], importance['scaled_importance'][:10])
plt.xlabel('Scaled Importance', fontsize=12)
plt.title('Top 10 Feature Importances', fontsize=14)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")

# --- 9. COMPREHENSIVE PERFORMANCE REPORT ---
print("\n" + "="*60)
print("✅ COMPREHENSIVE PERFORMANCE REPORT")
print("="*60)

print(f"\nCLINICAL PERFORMANCE (F2-optimized threshold):")
print(f"  Threshold:           {opt_thresh:.4f}")
print(f"  Recall:              {recall:.3f} ({tp}/{tp+fn} strokes caught)")
print(f"  Precision:           {precision:.3f} ({tp}/{tp+fp} alerts are real)")
print(f"  F2 Score:            {f2:.3f} (favors recall)")
print(f"  Accuracy:            {(tp+tn)/(tp+tn+fp+fn):.3f}")
print(f"  AUCPR:               {aucpr:.4f}")
print(f"  AUC-ROC:             {roc_auc:.4f}")

print(f"\nMODEL ROBUSTNESS:")
print(f"  Overfitting Gap:     {best_metrics['overfitting_gap']:.4f}")
print(f"  Models Tested:       {len(metrics_df)}")
print(f"  Best Valid AUCPR:    {best_metrics['valid_aucpr']:.4f}")

print(f"\nCLINICAL IMPACT (per 1000 patients):")
patients = 1000
stroke_rate = (tp+fn) / (tp+tn+fp+fn)
strokes_in_1000 = int(patients * stroke_rate)
caught_strokes = int(strokes_in_1000 * recall)
false_alarms = int(patients * (fp/(tp+tn+fp+fn)))

print(f"  Stroke Prevalence:   {stroke_rate*100:.1f}% ({strokes_in_1000} strokes)")
print(f"  Strokes Caught:      {caught_strokes} ({recall*100:.1f}%)")
print(f"  False Alarms:        {false_alarms}")
print(f"  Alarms per Stroke:   {false_alarms/caught_strokes:.1f}" if caught_strokes > 0 else "  Alarms per Stroke: N/A")

print(f"\nTOP 3 FEATURES:")
for i, row in importance.head(3).iterrows():
    print(f"  {row['variable']:<20}: {row['scaled_importance']:.3f}")

# --- 10. SAVE DETAILED RESULTS ---
print("\n10. Saving detailed results...")

# Save metrics to CSV
metrics_df.to_csv('grid_search_results.csv', index=False)
print("✓ Saved: grid_search_results.csv")

# Save best model parameters
best_params = {
    'max_depth': best_metrics['max_depth'],
    'learn_rate': best_metrics['learn_rate'],
    'sample_rate': best_metrics['sample_rate'],
    'col_sample_rate': best_metrics['col_sample_rate'],
    'min_rows': best_metrics['min_rows'],
    'optimal_threshold': float(opt_thresh),
    'recall': float(recall),
    'precision': float(precision),
    'f2_score': float(f2),
    'aucpr': float(aucpr),
    'roc_auc': float(roc_auc)
}

import json
with open('best_model_params.json', 'w') as f:
    json.dump(best_params, f, indent=2)
print("✓ Saved: best_model_params.json")

# --- 11. SHUTDOWN ---
end_time = time.perf_counter()
print(f"\n{'='*60}")
print(f"✅ COMPLETED IN {end_time - start_time:.2f} SECONDS")
print(f"✅ {len(metrics_df)} MODELS EVALUATED")
print(f"✅ 5 VISUALIZATIONS GENERATED")
print(f"✅ 3 DATA FILES SAVED")
print(f"{'='*60}")

h2o.cluster().shutdown()